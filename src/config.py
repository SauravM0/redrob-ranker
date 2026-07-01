"""Frozen reference date for the candidate pool.

Availability/recency and consistency checks are anchored to a FIXED snapshot date
rather than wall-clock `date.today()`, so the ranking is byte-for-byte reproducible
no matter what day it is run — the submitted CSV always matches the judges'
Stage-3 reproduction. This is the moment the dataset snapshot was taken.
"""
from datetime import date
SNAPSHOT_DATE = date(2026, 7, 1)
