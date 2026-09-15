"""Physics provenance, independent of simulator imports."""

PHYSICS_MODEL_VERSION = "rigid_point_payload_v2"
LEGACY_PHYSICS_MODEL_VERSION = "legacy_payload_v1"


def physics_comparison(training_version=None):
    """Missing provenance is unknown, never evidence of same-physics training."""
    return {
        "training_physics_model_version": training_version,
        "evaluation_physics_model_version": PHYSICS_MODEL_VERSION,
        "cross_physics_evaluation": (
            None
            if training_version is None
            else training_version != PHYSICS_MODEL_VERSION
        ),
        "physics_comparison_status": (
            "unknown_training_physics"
            if training_version is None
            else "same_physics"
            if training_version == PHYSICS_MODEL_VERSION
            else "cross_physics_evaluation"
        ),
    }


def manifest_physics_version(manifest):
    resolved = manifest.get("resolved_config")
    if not isinstance(resolved, dict):
        resolved = {}
    value = manifest.get(
        "physics_model_version",
        resolved.get("physics_model_version", LEGACY_PHYSICS_MODEL_VERSION),
    )
    return str(value)
