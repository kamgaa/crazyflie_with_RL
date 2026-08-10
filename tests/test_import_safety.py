from __future__ import annotations

import builtins
import importlib.util
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    "train_ppo.py",
    "train_ppo_02.py",
    "view_live.py",
    "view_live_hover.py",
    "circle_traj.py",
    "diag_entropy.py",
    "diag_iterm_sat.py",
    "plot_curve.py",
)
HEAVY_RUNTIME_IMPORTS = {
    "gymnasium",
    "matplotlib",
    "mujoco",
    "numpy",
    "stable_baselines3",
    "tensorboard",
    "torch",
}


@pytest.fixture(autouse=True)
def _disable_bytecode_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "dont_write_bytecode", True)


@pytest.mark.parametrize("filename", ENTRYPOINTS)
def test_entrypoint_import_is_lazy_and_has_no_side_effects(
    filename: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = ROOT / filename
    module_name = f"_import_safety_{source.stem}"
    original_import = builtins.__import__
    original_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "preserve-me")
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in HEAVY_RUNTIME_IMPORTS:
            raise AssertionError(f"{filename} imported heavyweight module {name!r}")
        return original_import(name, globals, locals, fromlist, level)

    def forbidden_write(*_args, **_kwargs):
        raise AssertionError(f"{filename} attempted a filesystem write while importing")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "mkdir", forbidden_write)
    monkeypatch.setattr(Path, "write_text", forbidden_write)
    try:
        spec = importlib.util.spec_from_file_location(module_name, source)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        assert callable(module.main)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "preserve-me"
    finally:
        sys.modules.pop(module_name, None)
        if original_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda


def test_evaluation_wrappers_map_to_the_preserved_default_profiles() -> None:
    expected = {
        "view_live.py": "e2e_hover_eval.yaml",
        "view_live_hover.py": "residual_circle_legacy006_eval.yaml",
        "circle_traj.py": "residual_circle_eval.yaml",
    }
    for filename, profile in expected.items():
        spec = importlib.util.spec_from_file_location(
            f"_profile_{Path(filename).stem}", ROOT / filename
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.DEFAULT_CONFIG.name == profile


def test_shared_evaluation_parser_exposes_required_compatibility_flags() -> None:
    from crazyflie_rl.eval_cli import build_parser

    parser = build_parser(ROOT / "configs" / "e2e_hover_eval.yaml", "test")
    destinations = {action.dest for action in parser._actions}
    assert {
        "config",
        "model",
        "headless",
        "policy",
        "preset",
        "no_realtime",
        "no_camera",
    } <= destinations
    args = parser.parse_args([])
    assert args.policy == "both"
    assert args.preset == "1"
    assert args.headless is False
    assert args.no_realtime is False
    assert args.no_camera is False


def test_entropy_diagnostic_does_not_infer_legacy_log_provenance() -> None:
    source = ROOT / "diag_entropy.py"
    spec = importlib.util.spec_from_file_location("_entropy_provenance", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.SOURCE_PROVENANCE.endswith("unverified")
