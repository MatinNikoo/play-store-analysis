"""playodds: odds that a new Google Play app reaches 10K, 100K or 1M installs, from launch-day facts."""

from playodds.model import comparables, load, score

__version__ = "0.2.0"
__all__ = ["score", "load", "comparables"]
