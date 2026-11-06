"""Public demo mode (``DEMO_MODE``): Iris only, no uploads, no deletions.

The Render demo runs master with ``DEMO_MODE=true``. These tests pin both
sides: with the flag on, visitors can neither add nor remove data and see only
the seeded Iris source; with it off (the default) nothing changes.
"""

import uuid
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from backend.config import Settings, get_settings
from backend.data_ingestion.demo import (
    IRIS_SOURCE_ID,
    seed_demo_iris,
)
from backend.database.models import DataSource, Deployment, TrainingJob
from backend.main import app

settings = get_settings()
engine = create_async_engine(f"sqlite+aiosqlite:///{settings.DB_PATH}", echo=False)
TestingSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
async def db_session(client):
    """Yield a session on the shared test database; ``client`` creates its tables."""
    async with TestingSessionLocal() as session:
        yield session


@pytest.fixture
def client():
    """Yield a client with the app lifespan running."""
    with TestClient(app, base_url="http://localhost") as c:
        yield c


@pytest.fixture
def demo_on(monkeypatch):
    """Turn demo mode on for one test; monkeypatch restores the default."""
    monkeypatch.setattr(get_settings(), "DEMO_MODE", True)


@pytest.fixture
def demo_off(monkeypatch):
    """Force demo mode off, even if the shell exports DEMO_MODE."""
    monkeypatch.setattr(get_settings(), "DEMO_MODE", False)


@pytest.fixture
async def iris(db_session, tmp_path) -> DataSource:
    """Seed the Iris demo source into the shared test database."""
    return await seed_demo_iris(db_session, tmp_path)


@pytest.fixture
async def other_source(db_session, tmp_path) -> DataSource:
    """Add an ordinary, successfully ingested source that demo mode must hide."""
    csv = tmp_path / "other.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    source = DataSource(
        source_id=f"other-{uuid.uuid4()}",
        name="Private sales data",
        type="file",
        config={"file_path": str(csv)},
        is_active=True,
        test_status="success",
    )
    db_session.add(source)
    await db_session.commit()
    await db_session.refresh(source)
    return source


def _source_ids(response) -> set[str]:
    """Collect the source_id of every source in a list response."""
    assert response.status_code == 200, response.text
    return {s["source_id"] for s in response.json()["sources"]}


def test_demo_mode_is_off_by_default():
    """The flag defaults to off so self-hosted installs keep every feature."""
    assert Settings.model_fields["DEMO_MODE"].default is False


def test_config_reports_demo_mode_off(client, demo_off):
    """The frontend reads the flag from /api/config; off by default."""
    assert client.get("/api/config").json()["demo_mode"] is False


def test_config_reports_demo_mode_on(client, demo_on):
    """With DEMO_MODE on, /api/config tells the frontend to hide upload UI."""
    assert client.get("/api/config").json()["demo_mode"] is True


def test_upload_rejected_in_demo_mode(client, demo_on):
    """Visitors cannot upload files to the shared demo server."""
    response = client.post(
        "/api/ingestion/upload", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")}
    )
    assert response.status_code == 403
    assert "demo" in response.json()["message"].lower()
    assert "Iris" in response.json()["message"]


def test_create_source_rejected_in_demo_mode(client, demo_on):
    """Visitors cannot register S3 sources on the shared demo server."""
    response = client.post(
        "/api/ingestion/database",
        json={"name": "S3", "type": "s3", "config": {"path": "s3://bucket/key.csv"}},
    )
    assert response.status_code == 403
    assert "demo" in response.json()["message"].lower()


async def test_delete_rejected_in_demo_mode(client, demo_on, iris, db_session):
    """One visitor must not delete the Iris source for everyone else."""
    response = client.delete(f"/data/api/sources/{iris.source_id}")
    assert response.status_code == 403
    assert await db_session.get(DataSource, iris.id) is not None


async def test_lists_show_only_iris_in_demo_mode(client, demo_on, iris, other_source):
    """Source lists, the usable list and the dataset filter show only Iris."""
    assert _source_ids(client.get("/data/api/sources")) == {IRIS_SOURCE_ID}
    assert _source_ids(client.get("/data/api/sources/usable")) == {IRIS_SOURCE_ID}
    datasets = client.get("/api/pipeline/datasets/list").json()
    assert [d["id"] for d in datasets] == [IRIS_SOURCE_ID]


async def test_other_sources_unreadable_in_demo_mode(client, demo_on, iris, other_source):
    """A non-Iris source is not found by id, sample, export, schema or EDA."""
    for path in (
        f"/data/api/sources/{other_source.source_id}",
        f"/data/api/sources/{other_source.id}/sample",
        f"/data/api/sources/{other_source.source_id}/export",
        f"/api/pipeline/datasets/{other_source.id}/schema",
        f"/api/eda/{other_source.id}/history",
    ):
        assert client.get(path).status_code == 404, path
    assert client.post(f"/api/eda/{other_source.id}/analyze").status_code == 404


async def test_stats_count_only_iris_in_demo_mode(client, demo_on, iris, other_source):
    """The dashboard counts only the Iris source while demo mode is on."""
    assert client.get("/api/pipeline/stats").json()["data_sources"] == 1


async def test_iris_readable_in_demo_mode(client, demo_on, iris):
    """The seeded Iris source stays fully usable in demo mode."""
    assert client.get(f"/data/api/sources/{IRIS_SOURCE_ID}").status_code == 200
    sample = client.get(f"/data/api/sources/{iris.id}/sample?limit=3")
    assert sample.status_code == 200
    assert len(sample.json()["data"]) == 3


@pytest.fixture
async def iris_job_under_hidden_jobs(db_session, iris):
    """One Iris job and deployment, sorted below two newer non-Iris ones.

    The future dates put these four rows at the top of every list, whatever
    else the shared test database holds.
    """
    tag = uuid.uuid4().hex[:8]
    rows = [
        ("iris", IRIS_SOURCE_ID, datetime(2099, 1, 1)),
        ("other1", "other", datetime(2099, 1, 2)),
        ("other2", "other", datetime(2099, 1, 3)),
    ]
    ids = []
    for name, dataset, when in rows:
        job_id = f"demo-{tag}-{name}"
        ids.append(job_id)
        db_session.add(
            TrainingJob(
                id=job_id,
                pipeline_id=job_id,
                node_id="model",
                dataset_source_id=dataset,
                status="completed",
                model_type=f"demo_{tag}_{name}",
                graph={},
                run_mode="fixed",
                started_at=when,
                created_at=when,
            )
        )
    await db_session.flush()
    for job_id, (name, _, when) in zip(ids, rows, strict=True):
        db_session.add(
            Deployment(
                job_id=job_id,
                model_type=f"demo_{tag}_{name}",
                artifact_uri="unused",
                is_active=False,
                created_at=when,
            )
        )
    await db_session.commit()
    yield ids[0]
    await db_session.execute(delete(Deployment).where(Deployment.job_id.in_(ids)))
    await db_session.execute(delete(TrainingJob).where(TrainingJob.id.in_(ids)))
    await db_session.commit()


@pytest.mark.parametrize(
    ("url", "job_of"),
    [
        ("/api/pipeline/jobs?limit=1", lambda row: row["job_id"]),
        ("/api/pipeline/jobs?limit=1&job_type=training", lambda row: row["job_id"]),
        ("/api/deployment/history?limit=1", lambda row: row["job_id"]),
        ("/api/registry/models?limit=1", lambda row: row["latest_version"]["job_id"]),
    ],
)
async def test_demo_lists_filter_before_paging(
    client, demo_on, iris_job_under_hidden_jobs, url, job_of
):
    """Hidden rows must not use up a page: the first page still shows the newest Iris row."""
    response = client.get(url)
    assert response.status_code == 200, response.text
    assert [job_of(row) for row in response.json()] == [iris_job_under_hidden_jobs]


async def test_lists_unchanged_when_demo_off(client, demo_off, other_source):
    """With demo mode off, every source stays listed and readable."""
    assert other_source.source_id in _source_ids(client.get("/data/api/sources"))
    assert client.get(f"/data/api/sources/{other_source.source_id}").status_code == 200


async def test_seed_is_idempotent(db_session, tmp_path):
    """Seeding twice keeps one Iris row and one CSV with all 150 rows."""
    first = await seed_demo_iris(db_session, tmp_path)
    second = await seed_demo_iris(db_session, tmp_path)
    count = await db_session.scalar(
        select(func.count(DataSource.id)).where(DataSource.source_id == IRIS_SOURCE_ID)
    )
    assert count == 1
    assert first.id == second.id
    csv = Path(second.config["file_path"])
    assert csv.exists()
    assert len(csv.read_text(encoding="utf-8").splitlines()) == 151
    assert second.test_status == "success"
    assert second.source_metadata["row_count"] == 150
    assert second.source_metadata["ingestion_status"]["status"] == "completed"


async def test_seed_repairs_stale_file_path(db_session, tmp_path):
    """Render's disk is wiped on redeploy; the seed rewrites the CSV and path."""
    await seed_demo_iris(db_session, tmp_path / "old")
    repaired = await seed_demo_iris(db_session, tmp_path / "new")
    assert Path(repaired.config["file_path"]).parent == (tmp_path / "new").absolute()
    assert Path(repaired.config["file_path"]).exists()
