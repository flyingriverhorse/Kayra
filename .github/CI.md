# CI execution

`pr_check.yml` is the push/PR entry point. It calls the backend, Core and
frontend workflows once. Each can also be run separately using
`workflow_dispatch`; those manual runs do not start SonarQube.

| Check | Runs per PR |
| --- | --- |
| Workflow syntax, expressions and reusable-workflow contracts (actionlint) | Once |
| Core suite + branch coverage (90% floor) | Once |
| Backend suite + branch coverage | Once per engine: pandas and Polars |
| Frontend unit tests + coverage | Once |
| Frontend lint/types, build/size and Playwright | Separate checks, preserved |
| Core Docker smoke | Install a wheel, train, predict, save/load |
| Backend Docker smoke | Registry wiring and serialization tests |
| SonarQube | Read coverage artifacts from the same workflow run |

The Polars backend report and the Core report also go to Codecov. Only the
copies downloaded by SonarQube have their Python filenames normalized.
Without `SONAR_TOKEN`, the scan is skipped; tests still run. A failed test
workflow blocks the scan, and available coverage remains in artifacts.

`docker-full-tests.yml` runs the existing Core unit/integration Docker suites
and backend unit Docker suite every Monday at 03:30 UTC. Use its manual
dispatch to reproduce container failures on a chosen branch. Local compose
commands retain their full suites.

## Required checks after migration

Reusable workflow jobs now appear under PR Check with `Backend Tests`,
`Skyulf-Core Tests` and `Frontend Tests` prefixes. After the first run, update
any branch protection or ruleset entries that refer to the previous standalone
check names. Remove the three old Sonar-only coverage job requirements; they
no longer run. Retain the normal test, complexity, Docker smoke, frontend
build/E2E and static-analysis requirements. Repository rules are not changed
by editing these workflow files.

Other workflows (Spark/Delta, security scans, docs and releases)
retain their own triggers.

CodeQL analyzes Python and JavaScript/TypeScript with `build-mode: none`;
it does not install application dependencies or run an autobuild. Spark/Delta
cancel older runs for the same ref when a new run starts.

The workflow validation gate uses actionlint 1.7.12 with a pinned SHA256.
It checks every workflow, including manual and scheduled ones. ShellCheck and
Pyflakes subprocess checks are disabled so this gate has the same scope locally
and in CI. Run `actionlint -shellcheck= -pyflakes=` from the repository root
to reproduce it. Include `Workflow validation (actionlint)` in required checks
if repository rules should enforce it before merge.
