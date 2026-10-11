"""Reproducible benchmarks for SecondChance.

``benchmarks.run`` is the entry point, run as ``./do bench``. It lives at the
repository root rather than under ``backend/`` because it benchmarks the
product, not the backend package: it imports ``app``
with ``backend/`` on ``PYTHONPATH`` (which the ``bench`` target sets) and keeps
its own fixtures and results next to it.

The rule that governs everything in here: a number that was not measured on
this machine, in this run, is not reported as a measurement. Anything the run
could not measure is emitted as an explicit ``NOT MEASURED`` entry that says
how to measure it, and nothing is ever extrapolated, rounded up from a vendor
datasheet or carried over from a previous run.
"""
