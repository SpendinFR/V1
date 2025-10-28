"""Light-weight helpers for metacognitive calibration tasks.

This package intentionally stays separate from :mod:`AGI_Evolutive.metacognition`
which implements the full metacognitive system.  Only generic utilities such as
calibration meters live here so that other modules can depend on them without
accidentally importing the much heavier ``metacognition`` package.  Re-exporting
the public helpers keeps import sites unambiguous::

    from AGI_Evolutive.metacog import CalibrationMeter

``metacognition`` continues to expose the agent-facing logic.
"""

from .calibration import CalibrationMeter, NoveltyDetector

__all__ = ["CalibrationMeter", "NoveltyDetector"]
