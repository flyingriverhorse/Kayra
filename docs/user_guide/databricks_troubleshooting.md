# Databricks project troubleshooting

Find the message from `smoke.py`, `preview.py` or the failed task below. The
index points to the page that owns each fix. The three detailed diagnostics at the
end come from the project validators. You can reproduce them locally without
Spark, credentials or a deployed job. Use the Core release matching your
generated project.

Each correction below is a change to an existing generated project, not a complete
replacement configuration. Start a project with the [walkthrough](databricks_bundle_walkthrough.md).
See the [glossary](databricks_glossary.md) for the lifecycle and data terms.

## Find your message

Search this table for the start of your message. Values in angle brackets vary.
The linked section explains the cause and the fix.

| Message or symptom | Where to fix it |
| --- | --- |
| `Pre-split step '<step>' reads target '<target>'. Set SKIP_TARGET_PRE_SPLIT_STEPS=True or SCORING_MODE='custom'.` | [Below](#a-pre-split-step-reads-the-target) |
| `feature_lookup.lookups[<n>].table_name: Unity Catalog names must be catalog.schema.name; each part a simple identifier (letters, digits, underscore; not starting with a digit).` | [Below](#a-feature-lookup-table-name-is-incomplete) |
| `feature_lookup.lookups[<n>] is not filled in: complete feature_lookup and online_features in config/training.yml, or regenerate with include_online_feature_lookup=false.` | [Below](#a-feature-lookup-table-name-is-incomplete) |
| `Automatic promotion requires quality_threshold.` | [Below](#automatic-promotion-has-no-quality-threshold) |
| `AliasOutcomeUnknownError`, or a pending alias event | [Lifecycle recovery](databricks_lifecycle_recovery.md#common-errors-and-fixes) |
| `Project and job definitions disagree; regenerate/redeploy the Bundle together.` after a Core or template upgrade (workflow contract) | [Project upgrade](databricks_project_upgrade.md#common-errors-and-fixes) |
| The same message after changing `follow_champion` in `config/inference.yml` | Redeploy the Bundle; see [score handoff](databricks_scoring.md#schedules-handoff-and-execution-engines) |
| `Training source has ... rows, exceeding max_rows=...`, `Training source observed size ... bytes exceeds max_bytes=...` or `Training source estimated size ... bytes exceeds max_bytes=...` | [Training data](databricks_training_data.md#common-errors-and-fixes): adjust `max_rows` / `max_input_mb` in `config/training.yml`, or set `training_sample_rows` to sample eligible rows before splitting. |
| `Training input has ... rows, exceeding max_rows=...`, `Training input size ... bytes exceeds max_bytes=...` or `Competition training data ... exceeds max_rows=... / max_bytes=...` | Python API: reduce the input size or adjust `max_rows` / `max_bytes` in `fit_workflow` or `evaluate_competition_candidate`. The frame training limit covers all splits together. Increase memory limits only when the process has enough memory. |
| `Initial scoring source exceeds the configured max_rows budget.` or `Source increment exceeds max_rows.` | [Scoring](databricks_scoring.md#common-errors-and-fixes) |
| `<node>.apply on pandas: <reason>`, for example `project_scoring.apply on pandas: Custom scoring and reused pre-split rules are unsupported.` (Spark admission) | [Distributed inference](databricks_spark_scoring.md) |
| An online endpoint returns `Encountered an unexpected error while evaluating the model. ... Error ''` | [Online features](databricks_online_features.md#check-missing-and-stale-features): read the served model logs |
| A Unity Catalog permission error in a job run (`PERMISSION_DENIED`, `INSUFFICIENT_PERMISSIONS`) | [Below](#permission-denied-in-a-job-run); test the job's `run_as` identity |
| `Set both monitoring_catalog and monitoring_schema.` | Set both in `deployment/variables.yml`, or leave both blank with `monitoring_enabled: "false"`. See [if no destination is set](databricks_monitoring_setup.md#if-no-destination-is-set) |
| `Local artifact runtime mismatch for skyulf-core: requires ..., found ...` or report reason `champion_runtime_unavailable` | The model was produced by another runtime. Approve a reviewed replacement with `allow_unloadable_champion=true`, or run with the producing release. See [below](#champion-saved-by-another-core-release). |
| `bundle init` fails with `ERROR_monitoring_schema_is_required_with_monitoring_catalog` | A monitoring catalog was answered without a schema. Answer both, or leave both blank to start without monitoring. See [project setup](databricks_project_setup.md#monitoring-destination). |
| `Monitoring cannot read training rows saved at runs:/...` or `... retrain it with the current project.` | [Training references](databricks_monitoring_enrollment.md#verify-training-references) |
| `Serving endpoint is not ready or its config update is pending.` | [Serving](databricks_serving.md#common-errors-and-fixes) |
| `Serving endpoint config update failed: UPDATE_FAILED.` | [Below](#serving-endpoint-update-failed) |
| `Owned serving endpoint must have exactly one served entity.` | [Below](#serving-endpoint-does-not-serve-one-entity) |
| `on_drift must be disabled or retrain.` | [Below](#on-drift-value-is-not-supported) |
| `Active performance_policy requires label_table and result_available_at_column.` | [Below](#performance-policy-without-a-label-table) |
| `Initialize the central monitoring store before reading.` (or `before writing.`) | [Below](#monitoring-store-is-not-initialized) |
| `STREAMING_TABLE_OPERATION_NOT_ALLOWED` | [Below](#streaming-table-operation-not-allowed) |

### Common errors on each topic page

| Topic | Typical symptoms |
| --- | --- |
| [Project setup](databricks_project_setup.md#common-errors-and-fixes) | Wrong schema in preview, missing columns or notebooks, monitoring destination |
| [Deployment](databricks_deployment.md#common-errors-and-fixes) | `REPLACE_*` values, authentication, privileges, duplicate jobs |
| [Project upgrade](databricks_project_upgrade.md#common-errors-and-fixes) | Graph contract, missing notebooks, saved model runtime |
| [Training data](databricks_training_data.md#common-errors-and-fixes) | Budgets, temporal boundaries, null labels, timezones |
| [Custom steps](databricks_custom_steps.md#common-errors-and-fixes) | Factory paths, row count, source package and dependencies |
| [Custom pre-split filters](databricks_custom_pre_split.md#common-errors-and-fixes) | Target-reading filters, non-Boolean masks |
| [Decision thresholds](databricks_decision_thresholds.md#common-errors-and-fixes) | Too few classes, unknown manual class, threshold conflicts |
| [Training reports](databricks_training_reports.md#common-errors-and-fixes) | Missing report tasks, charts or SHAP, truncated parameters |
| [Explanations](databricks_explanations.md#common-errors-and-fixes) | Missing SHAP tasks or artifacts, explanation status codes |
| [Model sets](databricks_model_sets.md#common-errors-and-fixes) | Replacement stays challenger, quality failures, Spark admission |
| [Approve, reject and roll back](databricks_lifecycle.md#common-errors-and-fixes) | Missing quality threshold, changed champion, scoring after promotion |
| [Lifecycle recovery](databricks_lifecycle_recovery.md#common-errors-and-fixes) | Uncertain alias changes |
| [Scoring](databricks_scoring.md#common-errors-and-fixes) | Champion selection, eligibility, budgets, schema |
| [Monitoring overview](databricks_monitoring.md#common-errors-and-fixes) | Monitoring destination, nothing monitored, old dashboard results |
| [Monitoring store setup](databricks_monitoring_setup.md#common-errors-and-fixes) | Store permissions and upgrades |
| [Monitoring enrollment](databricks_monitoring_enrollment.md#common-errors-and-fixes) | Enrollment and references |
| [Delayed labels](databricks_monitoring_labels.md#common-errors-and-fixes) | Label columns, timestamps, duplicate keys, classes |
| [Drift](databricks_drift.md#common-errors-and-fixes) | `no_data`, `degraded`, large PSI without drift |
| [Performance](databricks_performance.md#common-errors-and-fixes) | Unknown policy fields, missing baseline, failure streaks |
| [Retraining](databricks_retraining.md#common-errors-and-fixes) | Why no new train run appeared |
| [Monitoring dashboard](databricks_monitoring_dashboard.md#common-errors-and-fixes) | Issues to investigate, missing links |
| [Serving](databricks_serving.md#common-errors-and-fixes) | Endpoint readiness and privileges |
| [Online features](databricks_online_features.md#common-errors-and-fixes) | Missing and stale online features |
| [Serving monitoring](databricks_serving_monitoring.md#common-errors-and-fixes) | `awaiting_logs`, `failed`, blank counts or latency |
| [Daily rollout](databricks_serving_rollout.md#common-errors-and-fixes) | Retries, stops and recovery of traffic updates |
| [Remove a project](databricks_teardown.md#common-errors-and-fixes) | Unexpected deletions, shared dashboard, endpoints left running |

## Job task failures

A failed job task prints `FAILED`, the source line and, for the messages below,
a `Fix:` line and a `Docs:` link to this section. Other failures print
"Inspect the traceback below and this task's inputs before retrying."

### Serving endpoint update failed

**Cause.** Model Serving could not build or start the new served entity. The error
continues with `Served entity <name>: <state message>` and the tail of the build
and server logs. A failed build is usually a package install: a missing or
unreachable package, or a pin without a wheel for the serving platform.

**Fix.** Correct that cause, for example the requirement pin or the
[private package index](databricks_serving.md#operational-rules), then rerun
the same job. The job retries the same endpoint; you do not need to delete it.

### Serving endpoint does not serve one entity

**Cause.** The job manages an endpoint it owns only while that endpoint (and any
pending config) serves exactly one version of this model with all traffic. Someone
added or removed a served entity outside the job, for example in the Serving UI.

**Fix.** Restore a single served entity of this model with a 100 percent traffic
route, then rerun the job. To return to an earlier version, rerun with
`model_version=<previous_version>`. See [serving](databricks_serving.md).

### Champion saved by another Core release

**Cause.** A saved model loads only under the exact runtime that produced it. After
a Core upgrade the old champion cannot be loaded, so comparison reports
`champion_runtime_unavailable` and scoring it fails with
`Local artifact runtime mismatch`.

**Fix.** Review the candidate's own metrics in its training report, then run the
approve action the report prints with `allow_unloadable_champion: true`. No
retraining is needed. To keep using the old champion, run it with the Core release
that produced it. See [champion from an older Core release](databricks_lifecycle.md#champion-from-an-older-core-release).

### On drift value is not supported

**Cause.** The `on_drift` Bundle variable is neither `disabled` nor `retrain`.

**Fix.** Set `on_drift: disabled` or `on_drift: retrain` in
`deployment/variables.yml` and in every target override in
`deployment/targets.yml`, then redeploy. `smoke.py` checks both files. See
[retraining](databricks_retraining.md#what-to-edit).

### Performance policy without a label table

**Cause.** `monitoring_performance_policies` has a policy in `report` or `retrain`
mode, but no outcome table is bound.

**Fix.** Set `monitoring_label_table` and `monitoring_result_available_at_column`
for the target, or set `monitoring_performance_policies: '{}'` to turn
performance monitoring off. Redeploy. See [performance](databricks_performance.md#what-to-edit).

### Monitoring store is not initialized

**Cause.** The central monitoring tables do not exist yet in
`monitoring_catalog.monitoring_schema`, and this job only reads or writes them.

**Fix.** Have the store owner run
`initialize_monitoring_store(spark, monitoring_catalog, monitoring_schema)` once,
or give this job's identity the privileges to create them, then rerun. See
[monitoring store setup](databricks_monitoring_setup.md#the-command).

### Streaming table operation not allowed

**Cause.** A streaming table or materialized view has no Delta history, so its
version cannot be pinned (Databricks refuses `DESCRIBE HISTORY` and time travel on
them from dedicated compute).

**Fix.** Training reads such a `training_table` (or `training_label_table`)
directly: the run reads it once, saves the rows with the training run in MLflow,
and registration, comparison, approval and monitoring replay those saved rows. This
needs the generated job graph (`workflow_contract: 3`); a project generated by an
older Skyulf version stops with "only the generated job graph ... can train from
it", so regenerate it and redeploy. A features-job copy is no longer needed for
training. The batch score job, monitoring and `feature_lookup` (which stops with
"feature_lookup needs Delta history") still need a regular Delta table: read the
source through the features job and use its `output_table` there. See
[streaming tables as sources](databricks_features.md#lakeflow-streaming-tables-and-materialized-views-as-sources).

### Permission denied in a job run

**Cause.** The job's `run_as` identity lacks a Unity Catalog grant on a catalog,
schema, table, volume or model named in the error. The deployer's own access does
not count.

**Fix.** Grant the missing privilege to the `run_as` identity, then rerun. See
[data, model and experiment access](databricks_deployment.md#data-model-and-experiment-access).

<!-- diagnostic:target-pre-split -->
## A pre-split step reads the target

Example message, where the step name and target depend on your recipe:

```text
Pre-split step 'dropmissingrows' reads target 'churn'. Set SKIP_TARGET_PRE_SPLIT_STEPS=True or SCORING_MODE='custom'.
```

**Cause.** The selected `config/pre_split.yml` recipe filters or edits `churn`,
the actual target. That can be appropriate when choosing labeled training rows.
The default scoring mode also reuses this recipe, but prediction requests do not
contain the actual target. Project loading resolves the saved scoring policy,
so this can fail during the **training preview**, before fitting or scoring.

**Fix.** Decide the scoring policy in `config/scoring.yml` before preview
and training. To reuse the feature checks while omitting target-dependent rules,
set:

```yaml
version: 1
mode: pre_split
skip_target_pre_split_steps: true
eligibility: []
outputs: []
combined: []
```

A filter reading both target and feature columns is skipped **as a whole**. Split
it into separate target and feature checks if the feature check must still apply
to scoring. Fixed edits can instead be projected onto the selected input columns.

Alternatively, select `mode: custom` with `skip_target_pre_split_steps: false` and define the required eligibility
and output rules in the same file. Custom mode does not reuse pre-split rules;
the generated custom lists start empty. Do not select it expecting the commented
examples to become active. The training pre-split recipe still runs in either
case. These choices are saved with the newly fitted model; editing the file does
not change an already registered version's scoring policy.

**Check.** From the project root, run:

```bash
python src/tools/smoke.py
python src/tools/preview.py --action train
python src/tools/preview.py --action score
```

Confirm the selected recipe and scoring mode, then train a new model version.
Read [pre-split admission and scoring](preprocessing_recipes.md#pre-split-admission-and-scoring)
and [scoring eligibility](databricks_scoring.md#2-decide-eligibility-before-training-the-version).
<!-- /diagnostic -->

<!-- diagnostic:lookup-table -->
## A feature lookup table name is incomplete

Example message for the second lookup:

```text
feature_lookup.lookups[1].table_name: Unity Catalog names must be catalog.schema.name; each part a simple identifier (letters, digits, underscore; not starting with a digit).
```

**Cause.** `feature_lookup.lookups[N].table_name` is blank, contains an unresolved
placeholder, has fewer than three parts or has invalid identifier syntax. The index
is zero-based: `[0]` is the first list entry and `[1]` is the second. A completely
blank generated entry (no `table_name`, `lookup_key` or `feature_names`) reports
`feature_lookup.lookups[N] is not filled in: ...` instead: fill the entry, or
regenerate the project with `include_online_feature_lookup=false`. These checks validate names locally; passing them does not prove the table exists
or that the job can read it.

**Fix.** In `config/training.yml`, find the effective `feature_lookup` under
`defaults` or the model override and repair the indexed entry's `table_name`.
For example, replace an empty second table name with this field value:

```yaml
table_name: "{catalog}.{input_schema}.balances"
```

Here `catalog` and `input_schema` come from the selected project bindings. A
concrete `workspace.features.balances` also works. Use your own existing feature
table name and simple identifiers. Preserve the entry's `lookup_key`,
`feature_names` and any timestamp lookup settings; ensure they match the table's
keys and the model's declared inputs. A model override replaces the complete
lookup mapping, so check the effective model entry rather than repairing only a
shadowed default. The syntax correction does not create or merge a feature table.

**Check.** Run the three local checks above with the intended catalog/schema bindings. The expanded
table name should resolve without this message. Then verify table existence, feature metadata and
permissions in the chosen workspace before training. See
[feature tables and lookup choices](databricks_features.md) and the
[training reference](databricks_training_config.md#nested-settings-and-conditional-controls).
<!-- /diagnostic -->

<!-- diagnostic:automatic-threshold -->
## Automatic promotion has no quality threshold

```text
Automatic promotion requires quality_threshold.
```

**Cause.** The effective model configuration has `promotion_policy: automatic`
and a missing or `null` `quality_threshold`. A relative improvement requirement
or additional `quality_gates` does not replace the primary absolute threshold.
The validator stops before remote training and registry work.

**Fix.** In `config/training.yml`, set a numeric threshold under `defaults` or
the affected model entry, using the units and direction of its `metric`. For an
existing regression model, this is a fragment illustrating an RMSE upper bound:

```yaml
defaults:
  promotion_policy: automatic
  metric: heldout_rmse
  quality_threshold: 5.0
```

The value `5.0` is an example, not a recommended business limit. Choose your
acceptance criterion: lower RMSE is better, while classification accuracy/F1
needs a minimum acceptable score. Check model overrides for a `null` that masks the
default. Keep any additional gates and `min_improvement` requirements appropriate
to the same target and task. Choosing `manual_approval` allows a pending candidate
without this automatic-policy check; first-champion activation still needs an
explicit quality threshold and passing saved evidence.

**Check.** Run `smoke.py` and `preview.py --action train` again. Passing validation
only confirms a complete policy. After training, inspect the saved held-out metric
and every gate in `training_report` before accepting the candidate. Read the
[quality fields](databricks_training_config.md#quality-weights-and-bounded-reports)
and [registry lifecycle](mlflow_registry.md).
<!-- /diagnostic -->

## If your message is different

Keep the full traceback, task key, model/branch name, concrete model version and
the effective configuration for the failing action. Inspect the first failing
task; excluded conditional tasks are often expected. Retry only after checking
whether training, alias changes or prediction publication already committed.
[Approve, reject and roll back](databricks_lifecycle.md) explains saved results and
operator actions; [scoring recovery](databricks_scoring.md#recover-recognized-cdf-expiry)
covers the recognized CDF-expiry path. This page does not enumerate every validator
or imply that a local preview verifies cloud execution.
