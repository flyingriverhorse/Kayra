# Databricks notebook source
"""Score the pinned coherent model set using its saved component and rule code."""

from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task
from skyulf.integrations.databricks.model_sets.model_set_project import run_model_set_score_notebook
from skyulf.integrations.databricks.scoring.shared.prediction_output import grant_prediction_readers

if __name__ == "__main__":
    with notebook_task("score_models", globals()["dbutils"]):
        output = run_model_set_score_notebook(
            globals()["spark"],
            globals()["dbutils"],
            display_html=globals().get("displayHTML"),
            exit_notebook=False,
        )
        # Readers in access_data_readers (deployment/targets.yml) get SELECT.
        grant_prediction_readers(globals()["spark"], globals()["dbutils"], output)

# COMMAND ----------

if __name__ == "__main__":
    globals()["dbutils"].notebook.exit(output)
