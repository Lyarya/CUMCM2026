"""Q3 forecast preprocessing entry points."""

from src.problem3.forecast_data import build_official_canonical


def preprocess():
    """Build the read-only-derived canonical Attachment-3 table in memory."""

    return build_official_canonical()
