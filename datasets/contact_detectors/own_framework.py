"""
Contact detector: our own framework.

Thin wrapper around DualHandDetector — delegates all logic
(hover calibration, palm coordinate system, contact state machine,
still-hold detection) directly to it.
"""

import os
import sys
import numpy as np

# Ensure project root is importable regardless of CWD
_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from src.hand_track.dual_hand_detector import DualHandDetector
from .base import ContactDetectorBase


class OwnFrameworkDetector(ContactDetectorBase):
    name = "own_framework"
    needs_calibration = True

    def __init__(self):
        self._det = DualHandDetector()

    @property
    def hover_result(self):
        return self._det.hover_result

    def process(self, frame: np.ndarray) -> bool:
        return self._det.process(frame)

    def get_screen_position(self) -> tuple:
        return self._det.get_screen_position()

    def get_writing_position(self):
        return self._det.get_writing_position()

    def consume_still_hold_event(self) -> bool:
        return self._det.consume_still_hold_event()

    def reset(self):
        self._det.reset()
