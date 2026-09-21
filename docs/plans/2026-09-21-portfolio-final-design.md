# Portfolio Documentation Design

## Goal

Turn the repository landing page into a concise AI-application portfolio and provide a reproducible local demo runbook plus interview-ready project narration.

## Information architecture

The README leads with the project outcome, highlights, architecture, and limitations, then provides directory structure, prerequisites, quick start, API contract, privacy boundary, tests, and troubleshooting. Detailed operator steps live in `docs/demo-runbook.md`; one-minute and three-minute explanations plus interview prompts live in `docs/project-presentation.md`.

## Reproducibility boundary

The repository intentionally excludes PPMI-derived training CSV files and `model_bundle.joblib`. The export command therefore accepts explicit `--source-root` and `--artifact` paths. A user with authorized source files can reproduce the bundle; a visitor without those files can still review the architecture, code, tests, and API contract but cannot regenerate the research model.

## GitHub presentation

Use a Mermaid flowchart rather than a binary image so the architecture remains version-controlled and renders directly on GitHub. Keep the README bilingual only where technical identifiers require English; the narrative remains Chinese for the target audience.
