"""SecondChance backend application package.

Layout (see ``docs/INTERFACES.md`` for the authoritative ownership table):

* ``app.core``        - configuration, error types, logging. No domain logic.
* ``app.schemas``     - the frozen wire/domain contract shared by every module.
* ``app.db``          - SQLAlchemy declarative base and ORM models.
* ``app.retrieval``   - program repository and candidate retrieval.
* ``app.eligibility`` - the deterministic rule engine.
* ``app.ai``          - Gemini client, prompts, deterministic fallbacks.
* ``app.workflows``   - LangGraph state, topology and nodes.
* ``app.services``    - orchestration between the layers above.
* ``app.api``         - FastAPI application and versioned routes.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
