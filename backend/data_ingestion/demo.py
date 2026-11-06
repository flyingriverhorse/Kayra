"""Public demo mode: one pre-loaded Iris dataset, no uploads, no deletions.

``DEMO_MODE=true`` turns a normal deployment into the public demo. Visitors get
the bundled Iris dataset only: uploads, new sources and deletions return 403,
every other data source behaves as if it did not exist, and lists that name a
dataset (jobs, models, deployments, EDA reports) keep only Iris rows. With the
flag off, which is the default, every helper here is a no-op.
"""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import HTTPException
from sklearn.datasets import load_iris
from sqlalchemy import Select, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.data_ingestion.engine.profiler import DataProfiler
from backend.database.models import DataSource

IRIS_SOURCE_ID = "iris-demo"
IRIS_NAME = "Iris Flower Dataset"
DEMO_BLOCKED_MESSAGE = (
    "This public demo does not accept uploads, new data sources or deletions. "
    "Use the pre-loaded Iris dataset, or run Kayra yourself to work with your own data."
)


def forbid_in_demo_mode() -> None:
    """FastAPI dependency: reject the request with 403 when demo mode is on."""
    if get_settings().DEMO_MODE:
        raise HTTPException(status_code=403, detail=DEMO_BLOCKED_MESSAGE)


def demo_scope(query: Select) -> Select:
    """Restrict a query that selects from DataSource to Iris when demo mode is on."""
    if get_settings().DEMO_MODE:
        return query.where(DataSource.source_id == IRIS_SOURCE_ID)
    return query


def hidden_in_demo(source: DataSource | None) -> bool:
    """Return True when demo mode is on and ``source`` is not the Iris source."""
    return get_settings().DEMO_MODE and getattr(source, "source_id", None) != IRIS_SOURCE_ID


async def ensure_visible_in_demo(session: AsyncSession, dataset_id: int) -> None:
    """Answer 404 for any dataset other than Iris when demo mode is on.

    Queries only when demo mode is on, so the default path is unchanged.
    """
    if get_settings().DEMO_MODE and hidden_in_demo(await session.get(DataSource, dataset_id)):
        raise HTTPException(status_code=404, detail="Dataset not found")


async def demo_dataset_ids(session: AsyncSession) -> set[str] | None:
    """Return every id that names Iris (source_id and primary key), or None when off.

    Jobs store the dataset as whichever id the canvas sent, so both forms count.
    """
    if not get_settings().DEMO_MODE:
        return None
    pk = await session.scalar(select(DataSource.id).where(DataSource.source_id == IRIS_SOURCE_ID))
    return {IRIS_SOURCE_ID} if pk is None else {IRIS_SOURCE_ID, str(pk)}


async def keep_demo_rows[T](
    session: AsyncSession, rows: Sequence[T], dataset_id: Callable[[T], Any]
) -> Sequence[T]:
    """In demo mode, keep only rows whose dataset id names Iris; otherwise return ``rows``."""
    ids = await demo_dataset_ids(session)
    if ids is None:
        return rows
    return [row for row in rows if dataset_id(row) in ids]


def _iris_frame() -> pl.DataFrame:
    """Build Iris with readable species labels as the only target column."""
    iris = load_iris()
    frame = pl.DataFrame(iris.data, schema=list(iris.feature_names), orient="row")
    return frame.with_columns(
        pl.Series("species", [str(iris.target_names[i]) for i in iris.target])
    )


def _iris_metadata(frame: pl.DataFrame, csv_path: Path) -> dict[str, Any]:
    """Mirror the metadata a completed upload ingestion stores on its source."""
    profile = DataProfiler.profile(frame)
    return {
        "ingestion_status": {
            "status": "completed",
            "progress": 1.0,
            "updated_at": datetime.now(UTC).isoformat(),
        },
        "original_filename": csv_path.name,
        "file_size": csv_path.stat().st_size,
        "schema": {col: stats["type"] for col, stats in profile["columns"].items()},
        "row_count": profile["row_count"],
        "column_count": profile["column_count"],
        "profile": profile,
    }


async def seed_demo_iris(session: AsyncSession, upload_dir: Path) -> DataSource:
    """Write the Iris CSV and create or refresh its source row; safe to run on every start.

    The CSV is always rewritten and the row's path and metadata refreshed, because a
    host such as Render wipes its disk on redeploy while the database survives.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (upload_dir / f"{IRIS_SOURCE_ID}.csv").absolute()
    frame = _iris_frame()
    frame.write_csv(csv_path)

    source = await session.scalar(select(DataSource).where(DataSource.source_id == IRIS_SOURCE_ID))
    if source is None:
        source = DataSource(source_id=IRIS_SOURCE_ID)
        session.add(source)
    source.name = IRIS_NAME
    source.type = "file"
    source.config = {"file_path": str(csv_path)}
    source.description = "The classic Iris flower dataset (150 rows, 3 species)."
    source.is_active = True
    source.test_status = "success"
    source.source_metadata = _iris_metadata(frame, csv_path)
    try:
        await session.commit()
    except IntegrityError:
        # Another worker inserted the row first; refresh that row instead.
        await session.rollback()
        return await seed_demo_iris(session, upload_dir)
    await session.refresh(source)
    return source
