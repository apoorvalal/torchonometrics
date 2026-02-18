# API Documentation

This project uses [`pdoc`](https://pdoc.dev) to generate API documentation from
Python docstrings.

## Build

From the repository root:

```bash
uv sync --extra docs
bash docs/build_api_docs.sh
```

By default, docs are generated into `docs/api/`.
The build script enumerates the full `torchonometrics` module tree so subpackages
like `torchonometrics.choice` and `torchonometrics.choice.dynamic` are included.

To choose a different output directory:

```bash
bash docs/build_api_docs.sh tmp/pdoc-site
```
