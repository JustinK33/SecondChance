# INTERFACES.md - the frozen shared contract

Everything in this repository is built against the types and signatures below.
Read this file before writing code, and before changing anything in `backend/app/schemas/`, `backend/app/core/`, or `backend/app/db/base.py`.

The authority rules are simple.

- The Python in `backend/app/schemas/`, `backend/app/core/` and `backend/app/db/base.py` is the contract. This document describes it; where the two disagree, the code wins and this file is the bug.
- Every signature below is what the rest of the system *will call*. If you own the file, implement exactly that signature. If you need it changed, raise it as a handoff note instead of changing it unilaterally - a silent signature change costs a whole phase.
- Import shared types from the package, not the submodule: `from app.schemas import UserProfile, MatchStatus`.

---

## 1. Product invariants that the types enforce

These are not style preferences. Several of them are enforced structurally, so violating one is a validation error rather than a code review comment.

| Invariant | How it is enforced |
| --- | --- |
| The rule engine decides a match status; the model only writes language | `EligibilityEvaluation` validates `status == STATUS_BY_OUTCOME[outcome]`, and `ProgramMatch` validates `status is evaluation.status` |
| Statuses are only `POTENTIAL_MATCH` / `MORE_INFO_NEEDED` / `UNLIKELY_MATCH` | `MatchStatus` has exactly three members; the words "eligible" and "ineligible" appear nowhere in the vocabulary |
| Missing data never disqualifies anyone | Conditions resolve to `Trilean`; `UNKNOWN` maps to `MORE_INFO_NEEDED`, and an `UNKNOWN` evidence leaf must name the field it was missing |
| Nothing is fabricated | `FactStatement.source_ids` has `min_length=1`; `AssistanceProgram` requires sources and a `last_verified_at` to be `VERIFIED`; requirement bullets render only author-written `description` text |
| Unverified data stays out of eligibility claims | `AssistanceProgram.supports_eligibility_claims()` requires `VERIFIED` plus an approved rule set; `RuleSet.is_usable_for_matching()` requires `ReviewStatus.APPROVED` |
| We never ask for an SSN or banking credentials | `ProfileFlag` and `PROFILE_FIELD_NAMES` are closed sets with no such field, so there is nowhere to put one |
| No personal content is ever logged | `app.core.logging` redacts by key denylist, scrubs by value shape and truncates free text; `tests/unit/test_log_redaction.py` asserts on the real pipeline output |
| No implied agency affiliation | `AssistanceProgram.administering_org` is required and displayed next to every program name |

---

## 2. Directory ownership

One owner per path. Do not edit a file you do not own, even to fix something obvious - state it as a handoff note instead.

| Path | Owner | Contents |
| --- | --- | --- |
| `backend/app/__init__.py`, all `app/**/__init__.py` | Contract | Package docstrings only |
| `backend/app/core/config.py`, `core/errors.py`, `core/logging.py` | Contract | Settings, error taxonomy, redacting logger |
| `backend/app/schemas/*.py` | Contract | Every shared type |
| `backend/app/db/base.py` | Contract | Declarative base, naming convention, mixins |
| `docs/INTERFACES.md` | Contract | This file |
| `backend/tests/unit/test_schemas.py`, `tests/unit/test_log_redaction.py` | Contract | Contract and privacy tests |
| `backend/app/core/security.py`, `core/middleware.py` | Backend infra (A) | JWT verification, request context middleware |
| `backend/app/db/session.py`, `db/models/*.py` | Backend infra (A) | Engine, session factory, ORM models |
| `backend/alembic/**` | Backend infra (A) | Migrations, including RLS policies |
| `backend/app/api/**` | Backend infra (A) | FastAPI app, routes, dependencies, exception handlers |
| `backend/app/main.py` | Backend infra (A) | ASGI entry point and lifespan |
| `backend/app/workflows/**` | Workflows (B) | LangGraph state, graph, nodes, checkpointer wiring |
| `backend/app/ai/**` | AI (B) | Gemini client, prompts, deterministic fallback |
| `backend/app/retrieval/**` | Retrieval (C) | Repository, hybrid retrieval, citation verification |
| `backend/app/eligibility/**` | Retrieval (C) | Rule engine, FPL tables |
| `backend/data/programs/**` | Retrieval (C) | Curated program catalog and rule sets |
| `backend/app/services/**` | Services (D) | Orchestration between the layers |
| `backend/scripts/**` | Services (D) | Seed, smoke, OpenAPI export |
| `frontend/**` | Frontend (E) | Next.js application |
| `backend/tests/**` except the two files above, `benchmarks/**` | Quality (F) | Everything else under test |

---

## 3. Error envelope

Every non-2xx response body has exactly this shape, including errors raised inside FastAPI.

```json
{
  "error": {
    "code": "validation_error",
    "message": "Some of the information sent was not valid. Please check and try again.",
    "detail": { "field": "household_size" },
    "request_id": "0cd7e887a543be7f3bb3f046fbd6f51f"
  }
}
```

- `code` is an `ErrorCode` member and is part of the public contract. The frontend switches on it.
- `message` is plain-language and safe to render verbatim. It is never an upstream exception string.
- `detail` is optional, structured and user-safe. It never contains personal content or stack traces.
- `request_id` is the correlation id. Surface it in the UI so a user can quote it.

Types live in `app.core.errors`.

| Exception | HTTP | `ErrorCode` | `is_retryable` |
| --- | --- | --- | --- |
| `InvalidRequestError` | 422 | `validation_error` | no |
| `UnauthorizedError` | 401 | `unauthorized` | no |
| `ForbiddenError` | 403 | `forbidden` | no |
| `NotFoundError` | 404 | `not_found` | no |
| `ConflictError` | 409 | `conflict` | no |
| `RateLimitedError` | 429 | `rate_limited` | yes |
| `InvalidModelOutputError` | 502 | `invalid_model_output` | yes |
| `UpstreamUnavailableError` | 503 | `upstream_unavailable` | yes |
| `UpstreamTimeoutError` | 504 | `upstream_timeout` | yes |
| `WorkflowFailedError` | 500 | `workflow_failed` | yes |
| `InternalError` | 500 | `internal_error` | no |

Signatures the API layer implements against:

```python
AppError(message: str | None = None, *, detail: dict[str, Any] | None = None,
         cause: BaseException | None = None)
AppError.to_detail(request_id: str | None = None) -> ErrorDetail
AppError.to_response(request_id: str | None = None) -> ErrorResponse
RateLimitedError(message=None, *, retry_after_s: int | None = None, detail=None, cause=None)
error_code_for_status(status_code: int) -> ErrorCode
```

Handler rules, owned by the backend-infrastructure agent:

- `AppError` -> `exc.status_code`, body `exc.to_response(request_id)`.
- `RequestValidationError` -> 422, `validation_error`, `detail` built from the field locations only, never from the submitted values.
- `StarletteHTTPException` -> `error_code_for_status(exc.status_code)`.
- Anything else -> 500 `internal_error` with the class default message. Log `exc_info`; return nothing from it.

---

## 4. Configuration

`app.core.config` is the only module permitted to read the environment.

```python
get_settings() -> Settings          # lru_cache(maxsize=1)
Settings(_env_file=None, **overrides)   # construct directly in tests
```

Fields map to the variables in `.env.example`, case-insensitively (`APP_ENV` -> `app_env`). Every incoming string is stripped, because the repository-root `.env` is CRLF and an unstripped model name makes httpx raise `InvalidURL`.

Derived values you should use instead of re-deriving:

| Property | Value |
| --- | --- |
| `database_url_async` | `postgresql+asyncpg://...` for the SQLAlchemy async engine |
| `database_url_sync` | `postgresql+psycopg://...` for Alembic |
| `database_dsn` | plain `postgresql://...` libpq DSN, required by `AsyncPostgresSaver` |
| `test_database_url_async` / `_sync` / `test_database_dsn` | the same three for the test database |
| `extraction_model`, `explanation_model`, `chat_model` | per-stage model id, falling back to `gemini_model` |
| `gemini_timeout_ms` | timeout in **milliseconds** for `types.HttpOptions(timeout=...)` |
| `gemini_enabled` | whether a live model call is possible at all |
| `cors_origins` | exact allowed origins, never `"*"` |
| `is_local`, `is_production`, `debug` | environment predicates |
| `estimate_cost_usd(prompt_tokens, output_tokens)` | USD at the configured prices |

Startup fails closed if any of these hold:

- `AUTH_MODE=dev` with `APP_ENV != local`.
- `AUTH_MODE=supabase` with an empty `SUPABASE_JWT_SECRET`.
- `APP_ENV != local` with `CHECKPOINTER != postgres`, an empty `GEMINI_API_KEY`, or an empty `SUPABASE_URL`.
- `GEMINI_TIMEOUT_S * (GEMINI_MAX_RETRIES + 1) > WORKFLOW_TIMEOUT_S`.

---

## 5. Logging

`app.core.logging` is the privacy control. Use it for every application log line; never use the standard library `logging` module directly, because only this pipeline is redacted.

```python
configure_logging(settings: Settings, *, stream: TextIO | None = None) -> None
get_logger(name: str) -> FilteringBoundLogger
hash_user_id(user_id: str | uuid.UUID) -> str          # "u_<16 hex>"
new_request_id() -> str
bind_request_context(request_id: str, user_id_hash: str | None = None, **extra) -> None
bound_request_context(request_id: str, user_id_hash: str | None = None, **extra)  # context manager
clear_request_context() -> None
redact_event_dict(event_dict: EventDict, *, max_text_chars: int) -> EventDict
make_redaction_processor(max_text_chars: int) -> Processor
```

Rules:

- Log ids, hashes, enums, counts and durations. Never income, narratives, emails, addresses, tokens or raw user ids.
- `bind_request_context` hashes a raw-looking id defensively, but pass `hash_user_id(user_id)` explicitly.
- Use `bound_request_context` inside LangGraph nodes and background jobs so a parent context is restored on exit.
- Three redaction layers run on every event: the key denylist (`DENY_KEYS`, `DENY_KEY_FRAGMENTS`), value scrubbing (bearer tokens, JWTs, `AIza...` keys, Postgres URLs with passwords, emails, money amounts) and truncation to `settings.log_max_text_chars`. `exception` and `stack` are exempt from truncation only.
- Adding a field to the denylist is always safe. Removing one needs a review.

---

## 6. Schema inventory

Exact spellings. Import from `app.schemas`.

**Enums** (all `StrEnum`; the member value is the wire value)

| Enum | Members |
| --- | --- |
| `MatchStatus` | `POTENTIAL_MATCH`, `MORE_INFO_NEEDED`, `UNLIKELY_MATCH` |
| `Trilean` | `TRUE`, `FALSE`, `UNKNOWN` |
| `WorkflowStatus` | `QUEUED`, `RUNNING`, `AWAITING_INPUT`, `COMPLETED`, `FAILED`, `INTERRUPTED`, `CANCELLED` |
| `WorkflowPhase` | `PENDING`, `EXTRACT_PROFILE`, `VALIDATE_PROFILE`, `AWAIT_CLARIFICATION`, `RETRIEVE_PROGRAMS`, `VALIDATE_EVIDENCE`, `CATEGORIZE_NEEDS`, `EVALUATE_ELIGIBILITY`, `AWAIT_TARGETED_CLARIFICATION`, `GENERATE_EXPLANATIONS`, `BUILD_ACTION_PLAN`, `VERIFY_CITATIONS`, `MERGE_OUTPUTS`, `PERSIST_RESULTS`, `DONE` |
| `ProgramCategory` | `FOOD`, `HEALTHCARE`, `UTILITIES`, `HOUSING`, `UNIVERSITY_EMERGENCY_AID`, `COMMUNITY_SERVICES`, `CASH_TAX_CREDITS`, `CHILDCARE` |
| `NeedCategory` | `FOOD`, `HEALTHCARE`, `UTILITIES`, `HOUSING`, `CHILDCARE`, `TRANSPORTATION`, `EDUCATION_COSTS`, `INCOME_SUPPORT`, `INTERNET_PHONE`, `LEGAL_HELP`, `MENTAL_HEALTH`, `EMERGENCY_EXPENSE`, `OTHER`, `UNKNOWN` |
| `StudentStatus` | `NOT_A_STUDENT`, `UNDERGRAD_FULL_TIME`, `UNDERGRAD_PART_TIME`, `GRADUATE`, `OTHER`, `UNKNOWN` |
| `EmploymentStatus` | `EMPLOYED_FULL_TIME`, `EMPLOYED_PART_TIME`, `SELF_EMPLOYED`, `SEASONAL_OR_GIG`, `UNEMPLOYED`, `UNABLE_TO_WORK`, `RETIRED`, `NOT_IN_LABOR_FORCE`, `OTHER`, `UNKNOWN` |
| `HousingSituation` | `RENTING`, `OWN_HOME`, `LIVING_WITH_FAMILY`, `UNIVERSITY_HOUSING`, `TEMPORARY_OR_SHELTER`, `UNHOUSED`, `OTHER`, `UNKNOWN` |
| `IncomePeriod` | `MONTHLY`, `ANNUAL` - no `UNKNOWN`; the unknown case is `income_period is None` with no amounts |
| `VerificationStatus` | `VERIFIED`, `NEEDS_REVIEW`, `UNVERIFIED` |
| `SourceFreshness` | `FRESH` (<90d), `AGING` (90-365d), `STALE` (>365d) |
| `ActionItemKind` | `DOCUMENT`, `APPLY`, `CONFIRM_INFO`, `CONTACT`, `FOLLOW_UP`, `LEARN` |
| `ReviewStatus` | `DRAFT`, `PENDING_REVIEW`, `APPROVED`, `CHANGES_REQUESTED`, `REJECTED`, `SUPERSEDED` |
| `ClarificationInputKind` | `single_select`, `multi_select`, `number`, `range`, `text` (lowercase values) |
| `ConfidenceLevel` | `HIGH`, `MEDIUM`, `LOW`, `UNKNOWN` |
| `ProfileFlag` | lowercase values: `receives_snap`, `receives_wic`, `receives_medicaid`, `receives_tanf`, `receives_ssi`, `receives_ssdi`, `receives_unemployment`, `receives_free_school_meals`, `is_pregnant`, `has_children`, `is_veteran`, `is_disabled`, `is_senior`, `is_caregiver`, `is_rutgers_affiliated`, `has_utility_shutoff_notice`, `has_eviction_notice`, `filed_taxes_last_year` |
| `CategoricalProfileField` | `state`, `county`, `student_status`, `employment_status`, `housing_situation` |

Also in `app.schemas.enums`: `NEED_TO_PROGRAM_CATEGORIES: dict[NeedCategory, tuple[ProgramCategory, ...]]`.

**Models**

| Module | Models |
| --- | --- |
| `common` | `Page[T]`; aliases `Cents`, `ConfidenceScore`, `CountyName`, `HttpUrlStr`, `NonEmptyStr`, `PlainText`, `Slug`, `SourceId`, `StateCode`, `UtcDatetime` |
| `profile` | `HouseholdComposition`, `UserProfile`, `ProfileDraft`, `ProfileSnapshot`, `IntakeRequest`, `IntakeResponse`, `FieldEvidence`, `ExtractedFlag`, `ExtractedProfile`, `ClarificationOption`, `ClarificationQuestion`, `ClarificationQuestionSet`, `RangeValue`, `ClarificationAnswer` |
| `eligibility` | `GeographyCondition`, `StudentStatusCondition`, `HouseholdSizeCondition`, `IncomeFplCondition`, `IncomeAbsoluteCondition`, `CategoricalCondition`, `AgeCondition`, `FlagCondition`, `AllOf`, `AnyOf`, `Not`, `Condition`, `ConditionKind`, `ConditionEvidence`, `EligibilityEvaluation`, `RuleSet`, `RequirementBullet` |
| `program` | `ProgramSource`, `GeoScope`, `AssistanceProgram`, `ProgramSummary`, `ProgramDetail`, `RetrievedProgram`, `ProgramPage` |
| `matching` | `MatchExplanation`, `ProgramMatch` |
| `plan` | `ActionItemDraft`, `ActionItem`, `ActionItemPatch`, `SharedDocument`, `ActionPlanDraft`, `ActionPlan` |
| `workflow` | `UserFacingError`, `WorkflowCreateRequest`, `WorkflowCreateResponse`, `WorkflowRunSummary`, `WorkflowRunDetail`, `WorkflowResumeRequest`, `MatchRequest` |
| `assistant` | `ChatMessage`, `ChatRequest`, `Citation`, `FactStatement`, `ChatResponse`, `RejectedCitation`, `CitationVerificationReport` |
| `ai` | `PromptRef`, `ModelUsage`, `GeminiResult[T]`, `ModelCallSource` |
| `system` | `HealthResponse`, `DependencyStatus`, `ReadinessResponse`, `AccountDeletionReceipt` |
| `contracts` | `ProgramRepository`, `Retriever`, `EligibilityEngine`, `GeminiPort`, `CitationVerifier` |

**Helper functions on the contract**

```python
# common
utcnow() -> datetime                                   # timezone-aware UTC
slugify(value: str) -> str
program_uuid(slug: str) -> uuid.UUID                   # uuid5(PROGRAM_NAMESPACE, slug)
action_item_uuid(run_id: uuid.UUID, dedup_key: str) -> uuid.UUID
clarification_question_id(run_id: uuid.UUID, field: str) -> str

# profile
UserProfile.income_monthly_range_cents() -> tuple[int | None, int | None]
UserProfile.income_annual_range_cents() -> tuple[int | None, int | None]
UserProfile.has_income_information() -> bool
UserProfile.flag(name: ProfileFlag) -> bool | None     # None means never stated
UserProfile.is_unknown(field_name: str) -> bool
ProfileDraft.set_fields() -> dict[str, object]
ExtractedProfile.confidence_for(field_name: str) -> float | None
ExtractedProfile.source_span_for(field_name: str) -> str | None
ExtractedProfile.to_profile_fields(*, min_confidence: float = 0.0) -> dict[str, object]

# eligibility
resolve_all_of(outcomes: Iterable[Trilean]) -> Trilean
resolve_any_of(outcomes: Iterable[Trilean]) -> Trilean
resolve_not(outcome: Trilean) -> Trilean
status_for_outcome(outcome: Trilean) -> MatchStatus
render_requirements(root: Condition, extra_limitations: Sequence[str] = ()) -> list[RequirementBullet]
RuleSet.is_effective_on(when: date) -> bool
RuleSet.is_usable_for_matching(when: date) -> bool

# program
compute_freshness(retrieved_at: datetime, now: datetime | None = None) -> SourceFreshness
ProgramSource.current_freshness(now=None) -> SourceFreshness
ProgramSource.supports_eligibility_claim(now=None) -> bool
AssistanceProgram.supports_eligibility_claims() -> bool
AssistanceProgram.to_summary() -> ProgramSummary

# plan
make_dedup_key(kind: ActionItemKind, subject: str) -> str
group_shared_documents(items: list[ActionItem]) -> list[SharedDocument]
ActionPlan.completed_count -> int
ActionPlan.progress_fraction -> float

# workflow
progress_message_for(status: WorkflowStatus, phase: WorkflowPhase) -> str
WorkflowRunDetail.progress_for(status, phase) -> tuple[str, float]
WorkflowRunDetail.to_summary() -> WorkflowRunSummary
```

### Income is a range

`UserProfile` carries `income_min_cents`, `income_max_cents` and `income_period`. There is no single income field. Consequences every implementer must respect:

- An amount without a period is a validation error. Do not default to monthly.
- `income_min > income_max` is a validation error.
- `income_monthly_range_cents()` rounds **outward** - minimum floored, maximum ceilinged - so a rounding artefact can only widen the range. Widening turns a `TRUE` into an `UNKNOWN` (we ask), never into a `FALSE` (we wrongly discourage someone).
- The engine compares the whole range against a threshold: entirely under -> `TRUE`, entirely over -> `FALSE`, straddling -> `UNKNOWN`.

### The `dedup_key` derivation

`dedup_key == f"{kind.value.lower()}:{canonical_subject}"` where `canonical_subject = DOCUMENT_SUBJECT_ALIASES.get(slugify(subject), slugify(subject))`.

Always build it with `make_dedup_key(kind, subject)`, and pass the canonical requirement ("proof of address") as `subject`, not the display title. Only `DOCUMENT` items are grouped into `ActionPlan.shared_documents`, and only when two or more distinct programs ask for them. Add new synonyms to `DOCUMENT_SUBJECT_ALIASES`, never to a caller.

---

## 7. Module seams

The `Protocol` definitions in `app.schemas.contracts` are the full list. Depend on these, not on concrete classes.

Everything that touches the network or the database is `async`. `EligibilityEngine` is deliberately synchronous: it is a pure function of a profile and a rule set, retrieval hands it everything it needs, and marking it `async` would imply I/O that does not happen.

### `ProgramRepository` - implemented by `app/retrieval/repository.py`

```python
async def list_programs(self, *, categories: Sequence[ProgramCategory] | None = None,
                        state: str | None = None, county: str | None = None,
                        query: str | None = None,
                        verification_statuses: Sequence[VerificationStatus] | None = None,
                        limit: int = 20, offset: int = 0) -> ProgramPage: ...
async def get_program(self, program_id: uuid.UUID) -> ProgramDetail | None: ...
async def get_program_by_slug(self, slug: str) -> ProgramDetail | None: ...
async def get_programs_by_ids(self, program_ids: Sequence[uuid.UUID]) -> list[AssistanceProgram]: ...
async def get_active_ruleset(self, program_id: uuid.UUID, *, on: date | None = None) -> RuleSet | None: ...
async def get_sources(self, program_id: uuid.UUID, *, source_ids: Sequence[str] | None = None) -> list[ProgramSource]: ...
```

Concrete: `SqlProgramRepository(session: AsyncSession)` and `FixtureProgramRepository(programs: Sequence[AssistanceProgram], rulesets: Sequence[RuleSet])`. An instance is bound to one session and must not be shared across concurrent tasks.

`get_active_ruleset` returning `None` means "cannot assess", never "does not qualify".

### `Retriever` - implemented by `app/retrieval/hybrid.py`

```python
async def retrieve(self, profile: UserProfile, *, needs: Sequence[NeedCategory] | None = None,
                   limit: int = 20) -> list[RetrievedProgram]: ...
async def retrieve_for_programs(self, program_ids: Sequence[uuid.UUID]) -> list[RetrievedProgram]: ...
async def search_sources(self, query: str, *, program_id: uuid.UUID | None = None,
                         limit: int = 5) -> list[ProgramSource]: ...
```

Concrete: `HybridRetriever(repository: ProgramRepository)`. Results are ordered by descending `retrieval_score`. Each candidate arrives complete - program, approved rule set, sources - so the engine performs no I/O.

### `EligibilityEngine` - implemented by `app/eligibility/engine.py`

```python
def evaluate(self, profile: UserProfile, program: AssistanceProgram, ruleset: RuleSet,
             *, now: datetime | None = None) -> EligibilityEvaluation: ...
def evaluate_many(self, profile: UserProfile, candidates: Sequence[RetrievedProgram],
                  *, now: datetime | None = None) -> list[EligibilityEvaluation]: ...
def affected_programs(self, changed_fields: Sequence[str],
                      candidates: Sequence[RetrievedProgram]) -> list[uuid.UUID]: ...
```

Concrete: `RuleEngine(fpl: FplTable)`. Supporting surface in the same package:

```python
# app/eligibility/engine.py
def evaluate_condition(condition: Condition, profile: UserProfile, *, fpl: FplTable) -> ConditionEvidence

# app/eligibility/fpl.py
class FplTable:
    def monthly_threshold_cents(self, *, household_size: int, pct: float, year: int) -> int | None
    def annual_threshold_cents(self, *, household_size: int, pct: float, year: int) -> int | None
    def has_year(self, year: int) -> bool
```

`monthly_threshold_cents` returns `None` for an unknown year or an out-of-table household size, which must produce `UNKNOWN` with the missing field recorded - never a guessed extrapolation.

### `GeminiPort` - implemented by `app/ai/gemini.py` and `app/ai/fallback.py`

```python
async def extract_profile(self, narrative: str, *, prefilled: ProfileDraft | None = None,
                          request_id: str | None = None) -> GeminiResult[ExtractedProfile]: ...
async def propose_clarifications(self, profile: UserProfile, *, missing_fields: Sequence[str],
                                 candidates: Sequence[RetrievedProgram] = (),
                                 max_questions: int = 3,
                                 request_id: str | None = None) -> GeminiResult[ClarificationQuestionSet]: ...
async def explain_match(self, profile: UserProfile, program: AssistanceProgram,
                        evaluation: EligibilityEvaluation,
                        *, request_id: str | None = None) -> GeminiResult[MatchExplanation]: ...
async def draft_action_plan(self, profile: UserProfile, matches: Sequence[ProgramMatch],
                            *, request_id: str | None = None) -> GeminiResult[ActionPlanDraft]: ...
async def answer_question(self, request: ChatRequest, *, sources: Sequence[ProgramSource],
                          request_id: str | None = None) -> GeminiResult[ChatResponse]: ...
```

Concrete: `GeminiClient(settings: Settings, client: genai.Client | None = None)`, `DeterministicAi(settings: Settings)`, and the selector `build_gemini_port(settings: Settings) -> GeminiPort` in `app/ai/factory.py`.

The deterministic implementation is not a test double. The configured model is not reachable with this project's API key (see `docs/TOOLCHAIN.md`), so the whole product must be usable through `DeterministicAi`, and `GeminiResult.usage.source` records which path ran (`live` / `cache` / `fallback`). A benchmark or demo must never present fallback output as model output.

No method may return or influence a `MatchStatus`. `explain_match` receives the decided status and only words it.

### `CitationVerifier` - implemented by `app/retrieval/citations.py`

```python
async def verify(self, citations: Sequence[Citation],
                 *, known_sources: Sequence[ProgramSource] | None = None) -> CitationVerificationReport: ...
async def verify_response(self, response: ChatResponse,
                          *, known_sources: Sequence[ProgramSource] | None = None) -> CitationVerificationReport: ...
```

Concrete: `SourceCitationVerifier(repository: ProgramRepository)`. Verification is deterministic identity and substring matching - no model participates, because that would be asking the generator to grade itself. Rejection reasons are the `RejectedCitation.reason` literals: `unknown_source`, `excerpt_not_found`, `url_mismatch`, `stale_source`, `program_mismatch`.

### Services - `app/services/`

```python
# intake_service.py
async def extract(session: AsyncSession, user_id: str, request: IntakeRequest,
                  *, request_id: str) -> IntakeResponse

# profile_assembly.py  (pure, synchronous - owns the precedence policy)
def assemble_profile(extracted: ExtractedProfile | None, *, base: UserProfile | None = None,
                     prefilled: ProfileDraft | None = None,
                     answers: Sequence[ClarificationAnswer] = (),
                     min_confidence: float = 0.5) -> tuple[UserProfile, list[str]]
# returns (profile, changed_fields). Precedence, highest first:
#   clarification answers > prefilled > base > extracted.
# A skipped answer appends its field to unknown_fields.

# workflow_service.py
async def create_run(session, user_id: str, request: WorkflowCreateRequest,
                     *, request_id: str) -> WorkflowCreateResponse
async def get_run(session, user_id: str, run_id: uuid.UUID) -> WorkflowRunDetail
async def list_runs(session, user_id: str, *, limit: int = 20, offset: int = 0) -> Page[WorkflowRunSummary]
async def resume_run(session, user_id: str, run_id: uuid.UUID, request: WorkflowResumeRequest,
                     *, request_id: str) -> WorkflowRunDetail
async def cancel_run(session, user_id: str, run_id: uuid.UUID) -> WorkflowRunDetail
async def reevaluate(session, user_id: str, request: MatchRequest,
                     *, request_id: str) -> WorkflowCreateResponse

# plan_service.py
def build_plan(run_id: uuid.UUID, matches: Sequence[ProgramMatch], draft: ActionPlanDraft,
               *, version: int = 1, now: datetime | None = None) -> ActionPlan   # pure
async def get_plan(session, user_id: str, plan_id: uuid.UUID) -> ActionPlan
async def list_plans(session, user_id: str, *, limit: int = 20, offset: int = 0) -> Page[ActionPlan]
async def set_item_complete(session, user_id: str, item_id: uuid.UUID,
                            patch: ActionItemPatch) -> ActionItem

# assistant_service.py
async def answer(session, user_id: str, request: ChatRequest, *, request_id: str) -> ChatResponse

# account_service.py
async def delete_user_data(session, user_id: str) -> AccountDeletionReceipt
```

Every service takes `user_id` as a separate argument sourced from the verified token. A `user_id` in a request body is never trusted as authorisation, and `run_id` / `plan_id` / `item_id` ownership is checked server-side on every call.

### Backend infrastructure - `app/db/`, `app/core/security.py`, `app/api/`

```python
# app/db/session.py
def create_engine(settings: Settings) -> AsyncEngine
def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]
async def get_session() -> AsyncIterator[AsyncSession]        # FastAPI dependency

# app/core/security.py
@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    source: Literal["dev", "supabase"]
    @property
    def log_identity(self) -> str      # hash_user_id(self.user_id)

async def verify_token(token: str, settings: Settings) -> AuthenticatedUser

# app/api/deps.py
async def current_user(...) -> AuthenticatedUser              # FastAPI dependency
```

---

## 8. LangGraph state

Defined in `app/workflows/state.py` as a `TypedDict`, owned by the workflows agent.

The rule is one writer per field. A field written by more than one node in a parallel branch must declare a reducer; the only two that do are `limitations` and `model_usage`.

| Field | Type | Sole writer |
| --- | --- | --- |
| `run_id` | `uuid.UUID` | initial state (`workflow_service.create_run`) |
| `user_id` | `str` | initial state |
| `request_id` | `str` | initial state |
| `narrative` | `str` | initial state |
| `prefilled` | `ProfileDraft \| None` | initial state |
| `status` | `WorkflowStatus` | `workflow_service` (outside the graph) |
| `phase` | `WorkflowPhase` | every node sets its own phase on entry; last write wins |
| `extracted` | `ExtractedProfile \| None` | `extract_profile` |
| `profile` | `UserProfile \| None` | `validate_profile` |
| `profile_missing_fields` | `list[str]` | `validate_profile` |
| `clarification_questions` | `list[ClarificationQuestion]` | `await_clarification` |
| `clarification_answers` | `list[ClarificationAnswer]` | `apply_answers` (from the `Command(resume=...)` payload) |
| `candidates` | `list[RetrievedProgram]` | `retrieve_programs` |
| `evidence_warnings` | `list[str]` | `validate_evidence` |
| `need_categories` | `list[NeedCategory]` | `categorize_needs` |
| `evaluations` | `list[EligibilityEvaluation]` | `evaluate_eligibility` |
| `eligibility_missing_fields` | `list[str]` | `evaluate_eligibility` |
| `iteration` | `int` | `evaluate_eligibility` (bounded loop guard, max 2) |
| `explanations` | `dict[str, MatchExplanation]` keyed by `program_slug` | `generate_explanations` |
| `plan_draft` | `ActionPlanDraft \| None` | `build_action_plan` |
| `citation_report` | `CitationVerificationReport \| None` | `verify_citations` |
| `matches` | `list[ProgramMatch]` | `merge_outputs` |
| `plan` | `ActionPlan \| None` | `merge_outputs` |
| `limitations` | `Annotated[list[str], operator.add]` | any node, reducer-merged |
| `model_usage` | `Annotated[list[ModelUsage], operator.add]` | any node making a model call, reducer-merged |
| `error` | `UserFacingError \| None` | the graph's error handler |

Notes for the workflows agent.

- `profile_missing_fields` and `eligibility_missing_fields` are deliberately two fields rather than one, so each has a single writer and the two clarification passes cannot clobber each other.
- The first fan-out is `retrieve_programs` / `validate_evidence` / `categorize_needs`. They write three disjoint fields and so need no reducer.
- The second fan-out is `generate_explanations` / `build_action_plan` / `verify_citations`. Same: three disjoint fields. `merge_outputs` is the join that assembles `matches` and `plan`.
- `narrative`, `extracted`, `profile` and `clarification_answers` are personal content. They are never logged and never leave the checkpoint row for that run.
- Checkpointing uses `AsyncPostgresSaver` with `thread_id == str(run_id)`, so a thread is scoped to one run and one user. Resume with `Command(resume=...)` on the same `thread_id`.
- Per `docs/TOOLCHAIN.md`, an interrupted `ainvoke` returns normally with an extra `__interrupt__` key; it does not raise.

---

## 9. REST API

Base path `/api/v1`. Every route except `/healthz` and `/readyz` requires a bearer token and is scoped to the authenticated user.

| Method | Path | Request body | Success | Response body | Error codes |
| --- | --- | --- | --- | --- | --- |
| `POST` | `/api/v1/intake` | `IntakeRequest` | 200 | `IntakeResponse` | 401, 422, 429, 502, 503, 504 |
| `POST` | `/api/v1/workflows` | `WorkflowCreateRequest` | 202 | `WorkflowCreateResponse` | 401, 409, 422, 429 |
| `GET` | `/api/v1/workflows` | - | 200 | `Page[WorkflowRunSummary]` | 401, 422 |
| `GET` | `/api/v1/workflows/{run_id}` | - | 200 | `WorkflowRunDetail` | 401, 403, 404 |
| `POST` | `/api/v1/workflows/{run_id}/resume` | `WorkflowResumeRequest` | 200 | `WorkflowRunDetail` | 401, 403, 404, 409, 422 |
| `POST` | `/api/v1/workflows/{run_id}/cancel` | - | 200 | `WorkflowRunDetail` | 401, 403, 404, 409 |
| `POST` | `/api/v1/matches` | `MatchRequest` | 202 | `WorkflowCreateResponse` | 401, 403, 404, 422 |
| `GET` | `/api/v1/programs` | - | 200 | `ProgramPage` | 422 |
| `GET` | `/api/v1/programs/{program_id}` | - | 200 | `ProgramDetail` | 404 |
| `GET` | `/api/v1/plans` | - | 200 | `Page[ActionPlan]` | 401, 422 |
| `GET` | `/api/v1/plans/{plan_id}` | - | 200 | `ActionPlan` | 401, 403, 404 |
| `PATCH` | `/api/v1/action-items/{item_id}` | `ActionItemPatch` | 200 | `ActionItem` | 401, 403, 404, 422 |
| `POST` | `/api/v1/assistant/chat` | `ChatRequest` | 200 | `ChatResponse` | 401, 422, 429, 502, 503, 504 |
| `DELETE` | `/api/v1/account/data` | - | 200 | `AccountDeletionReceipt` | 401 |
| `GET` | `/healthz` | - | 200 | `HealthResponse` | - |
| `GET` | `/readyz` | - | 200 / 503 | `ReadinessResponse` | - |

Query parameters:

- `GET /api/v1/workflows`, `/plans`: `limit` (1-200, default 20), `offset` (>=0, default 0).
- `GET /api/v1/programs`: `category` (repeatable `ProgramCategory`), `state`, `county`, `q`, `verification_status` (repeatable), `limit`, `offset`.

Conventions:

- `POST /api/v1/workflows` returns 202 with a durable `run_id`; the client polls `GET /api/v1/workflows/{run_id}`. No HTTP request is ever held open for a model-backed workflow.
- An `idempotency_key` that matches an existing run for the same user returns 200 with `deduplicated: true` and the original run, not a second run. A key reused with a *different* body is 409 `conflict`.
- `POST .../resume` on a run that is not `AWAITING_INPUT` is 409 `conflict`.
- A run belonging to another user is 404, not 403, so the API does not confirm that the id exists.
- `/readyz` returns 503 with `ready: false` when a required dependency is down. `/healthz` never touches a dependency.
- `gemini` is a **non-required** dependency of readiness. An unreachable model degrades the wording, never the service - the rule engine decides every status and retrieval is deterministic - so `ready` stays `true` and the status code stays 200 even when `gemini.ok` is `false`. Only `database` can drain an instance.
- The `gemini` entry distinguishes *a key is configured* from *the model is reachable*, because those are different claims and the probe makes no network call (a billable `generateContent` on every load-balancer poll would time out exactly when the model is slow). Its three states:
  - `ok: false`, `degraded: true`, note "model access not provisioned" - no key. Evidence, from configuration.
  - `ok: false`, `degraded: true`, `failures: n` - a live call really failed and `DegradingGeminiPort` has latched into its cool-off. Evidence, from a real attempt, read off the port with no new call.
  - `ok: true`, `degraded: false`, `failures: 0`, note "configured; no live call attempted yet" - no evidence either way. Never reported as "configured" alone, which is what previously let a key whose every call failed report healthy.
- `DependencyStatus.degraded` and `.failures` are `null` for a dependency with no fallback path, and `note` never carries an upstream exception message, a model id or a status code: these endpoints are unauthenticated and must not be reconnaissance tools.
- `GET /api/v1/programs` is the only unauthenticated data route and returns catalog data only.

---

## 10. Database

Migrations are Alembic, owned by the backend-infrastructure agent. The declarative base, the constraint naming convention and the shared mixins are in `app/db/base.py`.

```python
from app.db.base import Base, TimestampMixin, SoftDeleteMixin, UuidPk, JsonDict
```

- `NAMING_CONVENTION` gives every index, unique constraint, check, foreign key and primary key a deterministic name, which is what makes a generated Alembic downgrade work. Every `CheckConstraint` must pass `name=...`.
- `Base.type_annotation_map` maps `uuid.UUID` -> `UUID`, `datetime` -> `TIMESTAMPTZ`, `dict[str, Any]`/`list[Any]` -> `JSONB`, `str` -> `TEXT`, `Decimal` -> `NUMERIC(12,4)`. Never declare a naive `TIMESTAMP` column.
- `Base.__repr__` prints the class name and primary key only. Do not override it to interpolate column values.
- `TimestampMixin` gives server-defaulted `created_at` / `updated_at`.
- `SoftDeleteMixin` is for catalog and audit tables. User-owned rows are hard-deleted by the account deletion flow, because a privacy promise has to be real. There is no global soft-delete filter; queries filter `deleted_at IS NULL` explicitly.

| Table | Columns | Keys and indexes |
| --- | --- | --- |
| `profiles` | `id`, `user_id`, `version`, `state`, `county`, `student_status`, `household_size`, `household_composition` (jsonb), `income_min_cents`, `income_max_cents`, `income_period`, `employment_status`, `housing_situation`, `needs` (jsonb), `age`, `flags` (jsonb), `unknown_fields` (jsonb), `extras` (jsonb), `created_at`, `updated_at` | pk `id`; uq `(user_id, version)`; ix `(user_id, created_at)` |
| `assistance_programs` | `id`, `slug`, `official_name`, `administering_org`, `category`, `description`, `plain_language_summary`, `who_it_helps` (jsonb), `geo_scope` (jsonb), `official_url`, `application_url`, `phone`, `keywords` (jsonb), `verification_status`, `last_verified_at`, `active_rule_version`, `search_vector` (tsvector, generated), `created_at`, `updated_at`, `deleted_at` | pk `id` = `uuid5(PROGRAM_NAMESPACE, slug)`; uq `slug`; ix `category`, `verification_status`; GIN `search_vector` |
| `program_sources` | `id` (text, the `SourceId`), `program_id`, `url`, `title`, `publisher`, `excerpt`, `retrieved_at`, `content_hash`, `freshness`, `created_at`, `updated_at` | pk `id`; fk `program_id` -> `assistance_programs.id` ON DELETE CASCADE; ix `program_id` |
| `program_rule_versions` | `id`, `program_id`, `version`, `effective_from` (date), `effective_to` (date, null), `root` (jsonb), `source_ids` (jsonb), `review_status`, `limitations` (jsonb), `reviewed_by`, `reviewed_at`, `created_at`, `updated_at` | pk `id`; uq `(program_id, version)`; fk `program_id`; ix `(program_id, review_status, effective_from)` |
| `workflow_runs` | `id`, `user_id`, `status`, `phase`, `idempotency_key`, `profile_id`, `narrative`, `limitations` (jsonb), `error` (jsonb, null), `started_at`, `completed_at`, `created_at`, `updated_at` | pk `id`; uq `(user_id, idempotency_key)` where not null; fk `profile_id` -> `profiles.id`; ix `(user_id, created_at)`, `status` |
| `program_matches` | `id`, `run_id`, `program_id`, `rule_version`, `status`, `outcome`, `score` (numeric), `evidence` (jsonb), `missing_fields` (jsonb), `limitations` (jsonb), `explanation` (jsonb, null), `evaluated_at`, `created_at` | pk `id`; uq `(run_id, program_id)`; fk `run_id` ON DELETE CASCADE, fk `program_id`; ix `(run_id, status)` |
| `action_plans` | `id`, `run_id`, `user_id`, `version`, `title`, `summary`, `shared_documents` (jsonb), `created_at`, `updated_at` | pk `id`; uq `(run_id, version)`; fk `run_id` ON DELETE CASCADE; ix `(user_id, created_at)` |
| `action_items` | `id`, `plan_id`, `program_id`, `program_name`, `kind`, `title`, `detail`, `url`, `sort_order`, `is_complete`, `completed_at`, `dedup_key`, `created_at`, `updated_at` | pk `id` = `action_item_uuid(run_id, dedup_key)`; uq `(plan_id, dedup_key)`; fk `plan_id` ON DELETE CASCADE; ix `(plan_id, sort_order)`; ck `completed_at IS NOT NULL = is_complete` |
| `model_calls` | `id`, `run_id` (null), `model`, `prompt_name`, `prompt_version`, `source`, `prompt_tokens`, `output_tokens`, `thought_tokens`, `total_tokens`, `latency_ms`, `attempts`, `estimated_cost_usd`, `created_at` | pk `id`; fk `run_id` ON DELETE SET NULL; ix `(run_id)`, `(created_at)`. Contains no user content. |
| `audit_events` | `id`, `user_id_hash`, `event_type`, `run_id`, `program_id`, `request_id`, `detail` (jsonb, non-personal), `created_at` | pk `id`; ix `(user_id_hash, created_at)`, `event_type`. Stores the hash from `hash_user_id`, never a raw id. |

LangGraph checkpoint tables - `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations` - are created by `await AsyncPostgresSaver.setup()`, not by Alembic. Alembic must not try to manage them; exclude them in `env.py`'s `include_object`.

### Row Level Security

User-owned tables are `profiles`, `workflow_runs`, `program_matches`, `action_plans`, `action_items` and `audit_events`. Each gets `ENABLE ROW LEVEL SECURITY` plus a policy restricting every operation to `user_id = auth.uid()::text`, reached through the owning row for `program_matches` and `action_items`.

Catalog tables - `assistance_programs`, `program_sources`, `program_rule_versions` - are public-read and service-role-write.

`model_calls` and the checkpoint tables are service-role only.

RLS is defined in migrations so it is version-controlled and testable. Supabase is not provisioned yet, so these policies are implemented and switchable but have not been verified against a live project. The local `AUTH_MODE=dev` path connects as the owner and therefore bypasses RLS; that is why `tests/security/` must assert cross-user isolation at the service layer as well.

---

## 11. Changing the contract

1. Open a handoff note describing the change and the callers it affects.
2. Change the code in `backend/app/schemas/` (or `core/`, or `db/base.py`) and this document in the same commit.
3. Add or update a case in `tests/unit/test_schemas.py`.
4. Run `ruff check .`, `ruff format --check .`, `mypy app` and `pytest tests/unit -q`. All four must be clean.
