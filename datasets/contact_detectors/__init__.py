"""
Registry of available contact detection methods.

To add a new method:
  1. Create datasets/contact_detectors/<name>.py with a class
     that subclasses ContactDetectorBase.
  2. Add an entry to REGISTRY below.
  3. If the method needs CLI arguments, handle them in test.py's main().
"""

from .base import ContactDetectorBase
from .own_framework import OwnFrameworkDetector
from .palmpad import PalmPadDetector

REGISTRY: dict[str, type[ContactDetectorBase]] = {
    "own_framework": OwnFrameworkDetector,
    "palmpad":       PalmPadDetector,
}


def build_detector(name: str, **kwargs) -> ContactDetectorBase:
    """Instantiate a contact detector by registry name."""
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown detector '{name}'. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**kwargs)
