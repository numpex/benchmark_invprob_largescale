from .data import (
    RadioBatch,
    RadioDataBundle,
    RadioDataConfig,
    RadioInterferometryDataset,
    build_radio_dataloaders,
    discover_radio_entries,
)

_PHYSICS_EXPORTS = {
    "DeepinvDirtyImager",
    "DirtyImagerConfig",
    "IdentityNoise",
    "MyRadioInterferometry",
    "RadioPhysicsResult",
    "create_radio_physics",
}


def __getattr__(name):
    """Load the optional FINUFFT-backed physics API only when requested."""
    if name not in _PHYSICS_EXPORTS:
        raise AttributeError(name)
    from . import physics

    return getattr(physics, name)


__all__ = [
    "DeepinvDirtyImager",
    "DirtyImagerConfig",
    "IdentityNoise",
    "MyRadioInterferometry",
    "RadioBatch",
    "RadioDataBundle",
    "RadioDataConfig",
    "RadioInterferometryDataset",
    "RadioPhysicsResult",
    "build_radio_dataloaders",
    "create_radio_physics",
    "discover_radio_entries",
]
