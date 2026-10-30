"""Import rules of the package layout (docs/design.md#dependency-rules), checked on the source."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import transcribe


PACKAGE = Path(transcribe.__file__).parent
ENGINES = {"av", "faster_whisper", "mlx", "mlx_whisper", "sherpa_onnx"}
PURE = (
    "domain",
    "languages",
    "pipeline.steps.attribution",
    "pipeline.steps.diarization.clustering",
    "pipeline.steps.output",
    "pipeline.steps.rendering",
)


def _modules() -> dict[str, Path]:
    modules = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE).with_suffix("").parts
        modules[".".join(part for part in parts if part != "__init__") or "__init__"] = path
    return modules


MODULES = _modules()


def _imports(path: Path, *, top_level_only: bool = False) -> set[str]:
    """Imported module names; `top_level_only` skips imports inside functions."""
    tree = ast.parse(path.read_text("utf-8"))
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if top_level_only and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return names


def _roots(names: set[str]) -> set[str]:
    return {name.split(".")[0] for name in names}


@pytest.mark.parametrize("module", [name for name in MODULES if name.split(".")[0] not in {"cli", "__main__"}])
def test_only_the_cli_uses_click(module: str) -> None:
    imports = _imports(MODULES[module])

    assert "click" not in _roots(imports)
    assert not {name for name in imports if name == "transcribe.cli" or name.startswith("transcribe.cli.")}


@pytest.mark.parametrize("module", PURE)
def test_pure_modules_import_no_engine(module: str) -> None:
    assert not _roots(_imports(MODULES[module])) & (ENGINES | {"click"})


@pytest.mark.parametrize("module", list(MODULES))
def test_engines_are_imported_lazily(module: str) -> None:
    assert not _roots(_imports(MODULES[module], top_level_only=True)) & ENGINES


@pytest.mark.parametrize("module", ["config.settings", "config.hardware", "models.manifest", "domain", "languages"])
def test_foundation_modules_do_not_depend_on_the_pipeline(module: str) -> None:
    imports = _imports(MODULES[module])

    assert not {name for name in imports if name.startswith(("transcribe.pipeline", "transcribe.cli"))}
