"""Print the API's OpenAPI document, byte for byte as the server serves it.

Run it with ``./do types``, or directly::

    python -m scripts.export_openapi                      # to stdout
    python -m scripts.export_openapi -o ../frontend/openapi.json

The document this prints is the contract the frontend compiles against:
``./do types`` writes it to ``frontend/openapi.json`` and ``npm run types``
turns that file into ``frontend/lib/api-types.gen.ts``. Both of those files are
committed, so every byte here lands in a reviewable diff, and three properties
matter more than brevity.

**It needs nothing but the code.** The schema comes from the real
:func:`app.main.create_app`, but no lifespan runs, so there is no database
connection, no checkpointer, no workflow runner and no model client. This runs
on a machine with no Postgres, no ``.env`` and no API key, which is what CI is.

Getting that was not free. Importing :mod:`app.main` builds a module-level
``app = create_app()`` for ``uvicorn app.main:app``, and that calls
:func:`~app.core.config.get_settings`, which validates the whole configuration
and refuses to boot without either ``AUTH_MODE=dev`` or a Supabase JWT secret.
An exporter that inherited that requirement would fail in CI for reasons that
have nothing to do with the schema. So the import happens inside
:func:`_hermetic_process_env`, which installs the minimum valid local
configuration first.

**The output is a pure function of the source tree.** Inside that same context
the ``.env`` file is detached from ``Settings`` and every variable ``Settings``
declares is removed from the process environment, so neither the repository's
real ``.env`` - which holds a live API key an export tool has no business
reading - nor a CI job's exported variables can reach the document. That is not
only hygiene: the document really does depend on configuration, because
``info.contact.url`` is ``frontend_url``. A developer whose ``.env`` points at a
deployed frontend would otherwise rewrite the committed artifact on every
``./do types`` run, and the diff would read as an API change. Exporting from
the field defaults means the artifact tracks the code and only the code.

**The bytes match the live endpoint.** Serialisation mirrors FastAPI's
``JSONResponse`` exactly - compact separators, ``ensure_ascii=False``, no
trailing newline - so this script and ``curl $API/openapi.json`` produce the
same file, and ``frontend/package.json``'s ``types`` and ``types:live`` scripts
cannot disagree about what the contract is.

Writing with ``--output`` is atomic: the document is rendered in full, written
to a temporary file beside the target and then renamed over it. A plain shell
redirect truncates the committed artifact to zero bytes *before* finding out
whether the export works, which turns an import error into a lost contract
file. That is not hypothetical - it is what the previous ``./do types`` did.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

from app.core.config import Settings, get_settings

__all__ = ["main", "render_openapi"]

#: Written into the environment so that importing :mod:`app.main` - which
#: builds an application at module scope - finds a configuration that validates
#: on a machine that has none. ``auth_mode=dev`` is the one combination that
#: needs no secret; it does not affect the document, because the security
#: scheme is declared by the route dependencies in either mode.
_BOOTSTRAP_ENV: Final[dict[str, str]] = {"APP_ENV": "local", "AUTH_MODE": "dev"}

#: Overrides applied on top of the field defaults for the exported document.
#: The empty API key means no code path reached from here could talk to Gemini
#: even if one tried.
_EXPORT_OVERRIDES: Final[dict[str, Any]] = {
    "app_env": "local",
    "auth_mode": "dev",
    "gemini_api_key": "",
}


@contextmanager
def _hermetic_process_env() -> Iterator[None]:
    """Make configuration come from the field defaults alone, then restore everything.

    Three sources have to be shut off, because pydantic-settings reads all of
    them: ``os.environ`` (every name ``Settings`` declares is removed, leaving
    ``PATH`` and friends alone so the interpreter keeps working), the ``.env``
    file named in ``Settings.model_config``, and the cached settings instance.

    The cache is cleared on the way out as well as on the way in. Clearing it
    on entry stops a previously loaded real configuration from being reused;
    clearing it on exit stops the hermetic one from leaking into a process that
    did not ask for it, which matters if these functions are ever called from a
    test rather than from ``__main__``.
    """
    declared = {name.upper() for name in Settings.model_fields}
    saved_env = {key: value for key, value in os.environ.items() if key.upper() in declared}
    saved_env_file = Settings.model_config.get("env_file")

    for key in saved_env:
        del os.environ[key]
    os.environ.update(_BOOTSTRAP_ENV)
    Settings.model_config["env_file"] = None
    get_settings.cache_clear()
    try:
        yield
    finally:
        Settings.model_config["env_file"] = saved_env_file
        for key in _BOOTSTRAP_ENV:
            os.environ.pop(key, None)
        os.environ.update(saved_env)
        get_settings.cache_clear()


def render_openapi() -> str:
    """The OpenAPI document as the server would serialise it, without a trailing newline."""
    with _hermetic_process_env():
        # Imported here, not at module scope: ``app.main`` instantiates an
        # application on import, which validates the configuration, so it has
        # to happen after the hermetic environment is in place.
        from app.main import create_app

        schema = create_app(Settings(_env_file=None, **_EXPORT_OVERRIDES)).openapi()
    return json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


#: Generated artifacts are world readable. ``mkstemp`` creates 0600 files,
#: which is right for a secret and wrong for a committed schema that a Docker
#: build or another user has to read.
_ARTIFACT_MODE: Final[int] = 0o644


def _write_atomically(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one step, or leave it exactly as it was."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.chmod(_ARTIFACT_MODE)
        # Atomic on POSIX: readers see either the old file or the whole new one.
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.export_openapi",
        description="Print the OpenAPI document the API serves, for frontend type generation.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        type=Path,
        help="Write to this file atomically instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    document = render_openapi()
    if args.output is None:
        # Deliberately not ``print``: a trailing newline would make the file
        # differ from what the endpoint serves.
        sys.stdout.write(document)
        sys.stdout.flush()
    else:
        _write_atomically(args.output, document)
        print(f"wrote {args.output} ({len(document.encode())} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
