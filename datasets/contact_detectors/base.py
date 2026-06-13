"""
Abstract base class for all contact detection methods used in Exp3.

Each method must implement process() and get_screen_position().
The base provides default no-op implementations for optional capabilities.

Implementing a new method:
    1. Create a new file in this package.
    2. Subclass ContactDetectorBase and set `name` (used as output folder).
    3. Register it in __init__.py.
"""

from abc import ABC, abstractmethod
import numpy as np


class ContactDetectorBase(ABC):
    # Unique short name — used as the output subfolder under Exp3/
    name: str

    # Whether this method requires the CALIB phase (hover anchor calibration).
    # Set to False for methods that need no geometric calibration.
    needs_calibration: bool = True

    # Expose hover calibration progress (HoverDetectResult or None).
    # Only inspected when needs_calibration=True and state == "CALIB".
    hover_result = None

    @abstractmethod
    def process(self, frame: np.ndarray) -> bool:
        """
        Process one camera frame.
        Returns is_writing (True = finger in contact).
        Must update internal state so the other getters return fresh values.
        """
        ...

    @abstractmethod
    def get_screen_position(self) -> tuple:
        """Return (x, y) pixel position of the writing fingertip (index finger tip)."""
        ...

    def get_writing_position(self):
        """
        Return (u, v) coordinates in the palm local frame, or None.
        Used for recording strokes in a hand-relative coordinate system.
        """
        return None

    def consume_still_hold_event(self) -> bool:
        """
        Returns True exactly once when a still-hold event fires (finger held
        stationary while in contact for ~1 s), then resets to False.
        Used to auto-advance trials without the user pressing SPACE.
        """
        return False

    def reset(self):
        """Reset internal state. Called on recalibration."""
        pass
