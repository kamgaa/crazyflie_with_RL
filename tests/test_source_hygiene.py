from __future__ import annotations

import ast
import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_python_sources_compile() -> None:
    for path in PROJECT_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")


def test_active_python_has_no_external_linux_absolute_paths() -> None:
    offenders = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if path == Path(__file__):
            continue
        if "/home/" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert offenders == []


def test_retired_residual_scale_is_not_active_python() -> None:
    offenders = []
    for path in PROJECT_ROOT.rglob("*.py"):
        if path == Path(__file__):
            continue
        if "0.006" in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(PROJECT_ROOT).as_posix())
    assert offenders == []


def test_training_entrypoint_has_only_guarded_module_level_calls() -> None:
    source = (PROJECT_ROOT / "train_ppo_02.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    unexpected = []
    for node in module.body:
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Assign,
                ast.AnnAssign,
            ),
        ):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            continue
        unexpected.append(type(node).__name__)
    assert unexpected == []


def test_importing_entrypoints_creates_no_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))

    import train_ppo
    import train_ppo_02

    importlib.reload(train_ppo)
    importlib.reload(train_ppo_02)
    assert set(tmp_path.rglob("*")) == before
