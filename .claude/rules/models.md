---
paths:
  - "src/transcribe/models/**"
  - "scripts/provision_models.py"
  - "tests/models/**"
---

# Models

`uv run transcribe`, `uv run pytest -m engine` and `scripts/benchmark.py`
read models from `~/Library/Application Support/transcribe/models/`
(`Settings.models_root`).

- `src/transcribe/models/model_manifest.json` is the single source of truth:
  name, license, source, revision, HTTPS URL and sha256 of every model file.
- `provision_models.py` downloads what is missing and verifies every hash.
  `--only NAME` limits it to one asset; `--models-root DIR` targets another
  directory (the benchmark then needs `--set models_root=DIR`).
- To add or change a model: set `sha256` to `null` in the manifest, run the
  script with `--record` on a trusted network, and commit the hashes.
- Runtime hash checks are off by default (`Settings.verify_model_hashes`).
