# Toolchain and verified API reference

Every fact in this file was produced by running `inspect.signature`, importing the module, or
executing a real call against the installed packages on 2026-10-10.
Do not "fix" anything here from memory - re-run the probe instead.

## Environment

| Thing | Value |
| --- | --- |
| Package manager | `uv` 0.13.0 at `~/.local/bin/uv` |
| Backend Python | CPython 3.12.15 (uv-managed) in `backend/.venv` |
| System `python3` | 3.14.4, **no pip, no ensurepip - never use it** |
| Node for frontend | v22.23.3 via nvm. Default `node` on PATH is v14 and breaks Next.js |
| Docker | 29.8.1, Compose v5.5.1 |
| GNU make | **not installed on this host** and there is no sudo - see note below |
| Postgres | `postgres:16` (server reports 16.15) on host port **5433** |

### Shell preamble

Backend:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run --project /mnt/d/Dev/Projects/SecondChance/backend <cmd>
# or
. /mnt/d/Dev/Projects/SecondChance/backend/.venv/bin/activate
```

Frontend - required on every command:

```bash
. ~/.nvm/nvm.sh && nvm use 22 >/dev/null
```

### `make` is missing on this host

There is no `make` binary and no passwordless sudo to install one, so the `Makefile`
could not be executed directly here. It was validated with GNU Make 4.4.1 in a container:

```bash
docker run --rm -v "$PWD:/w" -w /w alpine:3.21 \
  sh -c 'apk add --no-cache make bash >/dev/null && make help && make -n <target>'
```

All 22 targets parse and expand correctly, and the command inside each target was run
directly on the host. Until `make` is installed (`sudo apt install make`), run the
underlying commands from the `Makefile` by hand.

### Databases

```
postgresql://secondchance:secondchance@localhost:5433/secondchance        # app
postgresql://secondchance:secondchance@localhost:5433/secondchance_test   # pytest
```

`docker-compose.yml` only creates `secondchance`. The `secondchance_test` database is
created idempotently by `./do db-up`, which every db-dependent target depends on, so
`./do test-int` and `./do db-reset` both guarantee it exists. If you bring the container
up with a bare `docker compose up -d`, run `./do db-up` once afterwards.

Verified with asyncpg:

```
asyncpg OK -> PostgreSQL 16.15 (Debian 16.15-1.pgdg13+2) on x86_64-pc-linux-gnu
current_database: secondchance
test db OK -> secondchance_test
```

---

## langgraph 1.2.14 - verified imports

`langgraph` exposes no `__version__` attribute. Use
`importlib.metadata.version("langgraph")` if you need it.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command, Send, Interrupt, StateSnapshot
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
```

`interrupt`, `Command`, `Send` and `Interrupt` all live in **`langgraph.types`**.
`Send` and `Interrupt` are still re-exported from `langgraph.constants` but that path
raises `LangGraphDeprecatedSinceV10` - do not use it.
`START` / `END` are re-exported from `langgraph.graph` (canonical) and also from
`langgraph.constants` and `langgraph.func`.

Constant values: `START == "__start__"`, `END == "__end__"`.

### Signatures

```python
interrupt(value: Any, *, response_schema: dict[str, Any] | type | None = None) -> Any

Command(*, graph: str | None = None,
           update: Any | None = None,
           resume: dict[str, Any] | Any | None = None,
           goto: Send | Sequence[Send | N] | N = ())

Send(node: str, arg: Any, *, timeout: float | timedelta | TimeoutPolicy | None = None)

Interrupt   # dataclass, fields: value, id, response_schema

StateGraph(state_schema: type[StateT],
           context_schema: type[ContextT] | None = None,
           *, input_schema=None, output_schema=None)

StateGraph.add_node(node: str | StateNode, action: StateNode | None = None, *,
                    defer: bool = False, metadata=None, input_schema=None,
                    retry_policy: RetryPolicy | Sequence[RetryPolicy] | None = None,
                    cache_policy=None, error_handler=None,
                    destinations: dict[str, str] | tuple[str, ...] | None = None,
                    timeout=None, trace_policy=None) -> Self

StateGraph.add_edge(start_key: str | list[str], end_key: str) -> Self

StateGraph.add_conditional_edges(source: str,
                                 path: Callable[..., Hashable | Sequence[Hashable]] | ...,
                                 path_map: dict[Hashable, str] | list[str] | None = None) -> Self

StateGraph.compile(checkpointer: Checkpointer = None, *, cache=None, store=None,
                   interrupt_before=None, interrupt_after=None, debug=False,
                   name: str | None = None, transformers=None) -> CompiledStateGraph
```

Note: `compile(checkpointer=...)` is the **first positional** parameter, and
`StateGraph.__init__` takes `state_schema` positionally. The builder methods return
`Self`, so they chain.

### The interrupt protocol - verified by running a graph

An interrupted `ainvoke` **returns normally** (it does not raise). The result is the
current state dict plus an extra key:

```
result1 keys: ['val', 'answer', '__interrupt__']
__interrupt__ present: True   type: list
  element type: Interrupt   value: {'question': 'need more info', 'field': 'household_size'}
  id: 0cd7e887a543be7f3bb3f046fbd6f51f
```

So:

```python
result = await app.ainvoke(initial_state, config=cfg)
if (pending := result.get("__interrupt__")):
    first: Interrupt = pending[0]
    question = first.value          # whatever you passed to interrupt()
    interrupt_id = first.id         # str
```

Resume by invoking with a `Command` on the **same `thread_id`**:

```python
cfg = {"configurable": {"thread_id": "..."}}
result = await app.ainvoke(Command(resume={"household_size": 4}), config=cfg)
```

The value passed to `resume=` becomes the return value of the `interrupt()` call inside
the node, and the node re-runs from its start. Verified output:

```
result2: {'val': "finished:{'household_size': 4}", 'answer': "{'household_size': 4}"}
```

`await app.aget_state(cfg)` returns a `StateSnapshot`; `snapshot.next` is a tuple of the
node names still to run (empty `()` when the run is complete).

### Checkpointers

In-memory class name is **`InMemorySaver`**. `MemorySaver` exists in the same module but
is literally the same object (`MemorySaver is InMemorySaver` -> `True`), so prefer
`InMemorySaver`.

```python
from langgraph.checkpoint.memory import InMemorySaver
saver = InMemorySaver()
```

Postgres:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

AsyncPostgresSaver(conn, pipe: AsyncPipeline | None = None, serde: SerializerProtocol | None = None)

@classmethod
AsyncPostgresSaver.from_conn_string(conn_string: str, *, pipeline: bool = False,
                                    serde: SerializerProtocol | None = None)
    -> AsyncIterator[AsyncPostgresSaver]   # async context manager
```

There is **no `asetup()`**. The method is named **`setup()` and it is a coroutine** -
`inspect.iscoroutinefunction(AsyncPostgresSaver.setup)` is `True`, so you must `await` it:

```python
async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
    await saver.setup()          # creates/migrates the checkpoint tables, idempotent
    app = graph.compile(checkpointer=saver)
```

`from_conn_string` is an `@asynccontextmanager`, so the saver is only valid inside the
`async with` block. For a long-lived FastAPI app, open it in the lifespan handler and keep
the `AsyncExitStack`, or construct `AsyncPostgresSaver(pool)` with your own
`psycopg_pool.AsyncConnectionPool` and call `await saver.setup()` once.

The connection string must be a **psycopg/libpq** DSN (`postgresql://...`), not a
SQLAlchemy `postgresql+asyncpg://` URL.

Other async methods: `aget`, `aget_tuple`, `alist`, `aput`, `aput_writes`,
`adelete_thread`, `acopy_thread`, `adelete_for_runs`, `aprune`,
`aget_delta_channel_history`.

`setup()` creates these tables in the target database (verified against the live
container): `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`,
`checkpoint_migrations`.

A full interrupt/resume round trip was executed against the Docker Postgres:

```
pg interrupt keys: ['answer', '__interrupt__'] | value: {'q': 'x'}
pg resume: {'answer': 'resumed'}
checkpoint tables: ['checkpoint_blobs', 'checkpoint_migrations', 'checkpoint_writes', 'checkpoints']
```

---

## google-genai 2.29.0 - verified imports

```python
from google import genai
from google.genai import types, errors
```

`genai.__version__ == "2.29.0"`.

### Client

```python
genai.Client(*, enterprise=None, vertexai=None, api_key=None, credentials=None,
             project=None, location=None, debug_config=None,
             http_options: HttpOptions | HttpOptionsDict | None = None)
```

`genai.Client(api_key=...)` is correct. `client.aio` is an `AsyncClient`;
`client.aio.models` is an `AsyncModels`.

### The async generate call

```python
await client.aio.models.generate_content(
    *,
    model: str,
    contents: ...,                       # str is accepted
    config: types.GenerateContentConfig | GenerateContentConfigDict | None = None,
) -> types.GenerateContentResponse
```

All three parameters are **keyword-only** and the method is a true coroutine function.

### Structured output

`types.GenerateContentConfig` confirmed to have: `response_mime_type`, `response_schema`,
`response_json_schema`, `temperature`, `max_output_tokens`, `system_instruction`,
`thinking_config`, `http_options`, `top_p`, `candidate_count`, `seed`.

```python
config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=MyPydanticModel,     # a pydantic BaseModel subclass works directly
    temperature=0.0,
)
```

`response.parsed` is a real field on `GenerateContentResponse` and, when
`response_schema` is a `BaseModel` subclass, it holds an **instance of that model**.
Verified through the SDK's own parsing path:

```
parsed via _from_response: Probe(household_size=4, state='NJ') | isinstance Probe: True
```

`response.parsed` is `None` when no `response_schema` was supplied. Always treat it as
`Model | None` and fall back to validating `response.text` yourself.

`GenerateContentResponse` fields: `automatic_function_calling_history`, `candidates`,
`create_time`, `model_status`, `model_version`, `parsed`, `prompt_feedback`,
`response_id`, `sdk_http_response`, `usage_metadata`.

### Token usage - exact attribute names

`response.usage_metadata` is a `GenerateContentResponseUsageMetadata` with these fields
(snake_case on the Python object, camelCase on the wire):

```
cache_tokens_details
cached_content_token_count
candidates_token_count
candidates_tokens_details
prompt_token_count
prompt_tokens_details
thoughts_token_count
tool_use_prompt_token_count
tool_use_prompt_tokens_details
total_token_count
traffic_type
```

For cost accounting use `prompt_token_count` (input), `candidates_token_count` (output)
and `total_token_count`. All of them are `int | None`, so coalesce with `or 0`.
`thoughts_token_count` is billed as output on thinking models.

### Timeouts and errors

Per-request timeout goes through `http_options` and is in **milliseconds**:

```python
config = types.GenerateContentConfig(
    http_options=types.HttpOptions(timeout=30_000),   # ms
    ...
)
```

Exception hierarchy in `google.genai.errors`: `APIError` is the base, with
`ClientError` (4xx) and `ServerError` (5xx) subclasses, plus
`UnknownApiResponseError`, `FunctionInvocationError`, `UnknownFunctionCallArgumentError`,
`UnsupportedFunctionError`.

```python
APIError(code: int, response_json: Any, response=None)
```

Retry on `ServerError` and on `ClientError` with `e.code == 429`. Never retry other 4xx.

`types.FinishReason` is an enum - check `response.candidates[0].finish_reason` against
`types.FinishReason.MAX_TOKENS` to detect truncation before trusting `parsed`.

### Model availability - BLOCKING ISSUE

**No live Gemini call can currently succeed with this project's key.**
`client.models.list()` succeeds, so the key itself is valid, but every `generateContent`
attempt fails. Re-probed 2026-10-10 against every model id the SDK lists:

| Model | Result |
| --- | --- |
| `gemini-2.5-flash-lite` | `404 NOT_FOUND` - "no longer available to new users ... use models/gemini-3.5-flash-lite" |
| `gemini-2.5-flash` | `404 NOT_FOUND` - same retirement message, pointing at `models/gemini-3.8-flash` |
| `gemini-3.5-flash-lite` | `403 PERMISSION_DENIED` - "Your project has been denied access. Please contact support." |
| `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.1-flash-lite`, `gemini-3-flash-preview`, `gemini-flash-latest`, `gemini-flash-lite-latest`, `gemma-4-31b-it` | `403 PERMISSION_DENIED` - same message |
| `gemini-3.1-pro-preview`, `gemini-pro-latest` | "You exceeded your current quota, please check your plan and billing details" |

The 2.5 generation is retired for new projects and this project is authorised for nothing
in the 3.x generation. This is an account-side problem, not a code problem: it is resolved
in the Google Cloud / AI Studio project (access request or billing), and no code change
can work around it. Until it is resolved the deterministic fallback is the only language
path the product has.

`GEMINI_MODEL` and the per-stage overrides default to `gemini-3.5-flash-lite`
(`app/core/config.py`, `.env.example`); a local `backend/.env` still pinning a retired
`2.5` id is why a live call answers `404` rather than `403`. Both are the same blocker.

Consequences for the build:

- The Gemini client must be behind an interface with a deterministic fake implementation,
  and the whole product must be testable and demoable with that fake.
- All Gemini tests must be marked `live` and skipped unless `RUN_LIVE_GEMINI=1`, exactly
  as `pyproject.toml` already declares.
- `GEMINI_MODEL` must stay config-driven so it is a one-line change once access is granted.
- **A readiness probe must not report model access as healthy on the strength of a key
  string.** `GET /api/v1/ready` answered `{"name": "gemini", "ok": true, "note":
  "configured"}` while 100% of model calls were failing, which is exactly the outage the
  probe exists to surface. It now reports what the process has evidence for: unprobed,
  or degraded with a failure count read off `DegradingGeminiPort`. See section 9 of
  `docs/INTERFACES.md`.

### `.env` line endings

The root `.env` is CRLF. A naive reader hands `"gemini-2.5-flash-lite\r"` to the SDK and
httpx then raises `httpx.InvalidURL: Invalid non-printable ASCII character in URL, '\r'`.
Settings loading must strip surrounding whitespace from every string value.
The `backend/.env` and `frontend/.env.local` written by this step are LF.

---

## Reproducing these probes

```bash
export PATH="$HOME/.local/bin:$PATH"
cd /mnt/d/Dev/Projects/SecondChance/backend
.venv/bin/python - <<'PY'
import inspect
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
print(inspect.signature(AsyncPostgresSaver.from_conn_string))
print(inspect.iscoroutinefunction(AsyncPostgresSaver.setup))
PY
```
