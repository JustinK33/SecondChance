"""The import path ``docs/INTERFACES.md`` and ``app.schemas.contracts`` name.

The contract documents the :class:`~app.schemas.contracts.Retriever`
implementation as ``app.retrieval.hybrid.HybridRetriever``. The implementation
itself lives in :mod:`app.retrieval.search`, next to the scoring it is built
out of and next to :func:`~app.retrieval.search.search_programs`, which is the
entry point everything else actually calls - splitting the class away from its
score would have put the two halves of one explanation in two files.

So this module exists to keep the documented import working rather than to
hold logic. Importing from either path gets the same object::

    from app.retrieval.hybrid import HybridRetriever   # as the contract names it
    from app.retrieval.search import HybridRetriever   # where it is defined

The same reasoning applies to ``app.retrieval.repository``, which the contract
also names: the SQL ``ProgramRepository`` implementation turned out to belong
beside the ORM models, as ``app.db.repositories.programs.SqlProgramRepository``,
and that module's docstring explains why. There is no shim for it, because
nothing was ever written against the old path.
"""

from __future__ import annotations

from app.retrieval.search import HybridRetriever, RetrievalScore, ScoredProgram

__all__ = ["HybridRetriever", "RetrievalScore", "ScoredProgram"]
