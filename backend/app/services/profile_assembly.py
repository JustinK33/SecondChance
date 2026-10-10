"""The profile precedence policy, as a single public entry point.

The implementation lives in :mod:`app.workflows.profile_merge`, because the
graph's merge nodes were the first callers and the function is pure. This module
is the seam ``docs/INTERFACES.md`` names, so a caller outside the graph - the
intake endpoint, the profile editor - depends on ``app.services`` rather than
reaching into the workflow package.

Precedence, highest first: clarification answers, then prefilled values the
user typed, then an existing stored profile, then model extraction. The order
is not arbitrary. Anything the user said explicitly beats anything we inferred,
and a skipped answer is recorded as a deliberate unknown rather than left as a
gap we will ask about again.
"""

from __future__ import annotations

from app.workflows.profile_merge import answer_to_fields, assemble_profile

__all__ = ["answer_to_fields", "assemble_profile"]
