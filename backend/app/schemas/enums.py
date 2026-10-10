"""Closed vocabularies shared by the API, the database and the frontend.

Two rules apply to every enum here:

1. **Unknown must be representable.** Wherever a value can be genuinely unknown
   - because the user did not say, not because we failed to parse - there is an
   explicit ``UNKNOWN`` member. Guessing a value would turn a
   ``MORE_INFO_NEEDED`` into a false ``UNLIKELY_MATCH``, which is the single
   worst failure this product can have.
2. **Values are stable wire strings.** All of these are ``StrEnum``, so the
   member value is what lands in JSON and in Postgres. Add members freely;
   never rename or repurpose one.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = [
    "NEED_TO_PROGRAM_CATEGORIES",
    "ActionItemKind",
    "CategoricalProfileField",
    "ClarificationInputKind",
    "ConfidenceLevel",
    "EmploymentStatus",
    "HousingSituation",
    "IncomePeriod",
    "MatchStatus",
    "NeedCategory",
    "ProfileFlag",
    "ProgramCategory",
    "ReviewStatus",
    "SourceFreshness",
    "StudentStatus",
    "Trilean",
    "VerificationStatus",
    "WorkflowPhase",
    "WorkflowStatus",
]


class MatchStatus(StrEnum):
    """The only three verdicts this product may ever express.

    "Eligible" and "ineligible" are determinations only the administering
    agency can make. SecondChance reports a preliminary assessment, so the
    vocabulary is deliberately hedged and these three values are exhaustive.
    """

    POTENTIAL_MATCH = "POTENTIAL_MATCH"
    MORE_INFO_NEEDED = "MORE_INFO_NEEDED"
    UNLIKELY_MATCH = "UNLIKELY_MATCH"


class Trilean(StrEnum):
    """Three-valued logic used by the rule engine.

    ``UNKNOWN`` is a first-class outcome, not an error: it means the profile
    does not yet contain the field the condition needs.
    """

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class WorkflowStatus(StrEnum):
    """Lifecycle of one workflow run.

    ``AWAITING_INPUT``  - parked on a LangGraph interrupt, resumable by the user.
    ``INTERRUPTED``     - the process died mid-run; resumable from the last
                          checkpoint without re-asking the user anything.
    ``FAILED``          - terminal, carries a user-safe error.
    """

    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    AWAITING_INPUT = "AWAITING_INPUT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCELLED = "CANCELLED"


class WorkflowPhase(StrEnum):
    """One member per LangGraph node, used to drive user-facing progress.

    The member names mirror the graph node names exactly so a node can report
    its own phase without a lookup table. Users never see these strings - see
    ``app.schemas.workflow.PHASE_PROGRESS_MESSAGES`` for the copy.
    """

    PENDING = "PENDING"
    EXTRACT_PROFILE = "EXTRACT_PROFILE"
    VALIDATE_PROFILE = "VALIDATE_PROFILE"
    AWAIT_CLARIFICATION = "AWAIT_CLARIFICATION"
    RETRIEVE_PROGRAMS = "RETRIEVE_PROGRAMS"
    VALIDATE_EVIDENCE = "VALIDATE_EVIDENCE"
    CATEGORIZE_NEEDS = "CATEGORIZE_NEEDS"
    EVALUATE_ELIGIBILITY = "EVALUATE_ELIGIBILITY"
    AWAIT_TARGETED_CLARIFICATION = "AWAIT_TARGETED_CLARIFICATION"
    GENERATE_EXPLANATIONS = "GENERATE_EXPLANATIONS"
    BUILD_ACTION_PLAN = "BUILD_ACTION_PLAN"
    VERIFY_CITATIONS = "VERIFY_CITATIONS"
    MERGE_OUTPUTS = "MERGE_OUTPUTS"
    PERSIST_RESULTS = "PERSIST_RESULTS"
    DONE = "DONE"


class ProgramCategory(StrEnum):
    """Category of an assistance program, as the catalog classifies it."""

    FOOD = "FOOD"
    HEALTHCARE = "HEALTHCARE"
    UTILITIES = "UTILITIES"
    HOUSING = "HOUSING"
    UNIVERSITY_EMERGENCY_AID = "UNIVERSITY_EMERGENCY_AID"
    COMMUNITY_SERVICES = "COMMUNITY_SERVICES"
    CASH_TAX_CREDITS = "CASH_TAX_CREDITS"
    CHILDCARE = "CHILDCARE"


class NeedCategory(StrEnum):
    """What the user said they need, in their own framing.

    Kept separate from :class:`ProgramCategory` because a need is a statement
    about a person and a category is a property of a program; the mapping
    between them is many-to-many and lives in
    :data:`NEED_TO_PROGRAM_CATEGORIES`.
    """

    FOOD = "FOOD"
    HEALTHCARE = "HEALTHCARE"
    UTILITIES = "UTILITIES"
    HOUSING = "HOUSING"
    CHILDCARE = "CHILDCARE"
    TRANSPORTATION = "TRANSPORTATION"
    EDUCATION_COSTS = "EDUCATION_COSTS"
    INCOME_SUPPORT = "INCOME_SUPPORT"
    INTERNET_PHONE = "INTERNET_PHONE"
    LEGAL_HELP = "LEGAL_HELP"
    MENTAL_HEALTH = "MENTAL_HEALTH"
    EMERGENCY_EXPENSE = "EMERGENCY_EXPENSE"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class StudentStatus(StrEnum):
    NOT_A_STUDENT = "NOT_A_STUDENT"
    UNDERGRAD_FULL_TIME = "UNDERGRAD_FULL_TIME"
    UNDERGRAD_PART_TIME = "UNDERGRAD_PART_TIME"
    GRADUATE = "GRADUATE"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class EmploymentStatus(StrEnum):
    EMPLOYED_FULL_TIME = "EMPLOYED_FULL_TIME"
    EMPLOYED_PART_TIME = "EMPLOYED_PART_TIME"
    SELF_EMPLOYED = "SELF_EMPLOYED"
    SEASONAL_OR_GIG = "SEASONAL_OR_GIG"
    UNEMPLOYED = "UNEMPLOYED"
    UNABLE_TO_WORK = "UNABLE_TO_WORK"
    RETIRED = "RETIRED"
    NOT_IN_LABOR_FORCE = "NOT_IN_LABOR_FORCE"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class HousingSituation(StrEnum):
    RENTING = "RENTING"
    OWN_HOME = "OWN_HOME"
    LIVING_WITH_FAMILY = "LIVING_WITH_FAMILY"
    UNIVERSITY_HOUSING = "UNIVERSITY_HOUSING"
    TEMPORARY_OR_SHELTER = "TEMPORARY_OR_SHELTER"
    UNHOUSED = "UNHOUSED"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class IncomePeriod(StrEnum):
    """The period an income figure was reported for.

    There is no ``UNKNOWN`` member on purpose. An income amount without a known
    period cannot be normalised, so the unknown case is modelled by leaving
    ``income_period`` unset - and ``UserProfile`` rejects an amount without one
    rather than assuming monthly.
    """

    MONTHLY = "MONTHLY"
    ANNUAL = "ANNUAL"


class VerificationStatus(StrEnum):
    """How much a catalog record has been checked by a human.

    Only ``VERIFIED`` records may support an eligibility claim. ``UNVERIFIED``
    records are shown as informational leads with an explicit label, which is
    how the product honours "never fabricate a requirement".
    """

    VERIFIED = "VERIFIED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    UNVERIFIED = "UNVERIFIED"


class SourceFreshness(StrEnum):
    """Age bucket of a retrieved official source document.

    ``FRESH``  - retrieved within 90 days.
    ``AGING``  - 90 to 365 days; shown with a "last verified" caveat.
    ``STALE``  - over 365 days; excluded from eligibility claims.
    """

    FRESH = "FRESH"
    AGING = "AGING"
    STALE = "STALE"


class ActionItemKind(StrEnum):
    """What kind of work one action item asks of the user."""

    DOCUMENT = "DOCUMENT"
    APPLY = "APPLY"
    CONFIRM_INFO = "CONFIRM_INFO"
    CONTACT = "CONTACT"
    FOLLOW_UP = "FOLLOW_UP"
    LEARN = "LEARN"


class ReviewStatus(StrEnum):
    """Human-review state of an extracted or edited rule set.

    A rule set only influences matching at ``APPROVED``. Everything else is
    visible to maintainers but inert, so a newly ingested requirement cannot
    change anyone's results before a person has read it.
    """

    DRAFT = "DRAFT"
    PENDING_REVIEW = "PENDING_REVIEW"
    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class ClarificationInputKind(StrEnum):
    """The control the frontend should render for a clarification question."""

    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"
    NUMBER = "number"
    RANGE = "range"
    TEXT = "text"


class ConfidenceLevel(StrEnum):
    """Coarse confidence band shown to users.

    Deliberately coarse: a numeric score implies a calibration this system does
    not have.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class ProfileFlag(StrEnum):
    """Closed set of boolean profile facts a rule may test.

    A flag is only ever set from something the user stated explicitly. Nothing
    here is inferred: a narrative mentioning a child does not set
    ``HAS_CHILDREN``; it produces a clarification question instead. Keeping the
    set closed is what makes that reviewable, and it keeps the extraction
    model's structured output constrained to names the rule engine understands.
    """

    RECEIVES_SNAP = "receives_snap"
    RECEIVES_WIC = "receives_wic"
    RECEIVES_MEDICAID = "receives_medicaid"
    RECEIVES_TANF = "receives_tanf"
    RECEIVES_SSI = "receives_ssi"
    RECEIVES_SSDI = "receives_ssdi"
    RECEIVES_UNEMPLOYMENT = "receives_unemployment"
    RECEIVES_FREE_SCHOOL_MEALS = "receives_free_school_meals"
    IS_PREGNANT = "is_pregnant"
    HAS_CHILDREN = "has_children"
    IS_VETERAN = "is_veteran"
    IS_DISABLED = "is_disabled"
    IS_SENIOR = "is_senior"
    IS_CAREGIVER = "is_caregiver"
    IS_RUTGERS_AFFILIATED = "is_rutgers_affiliated"
    HAS_UTILITY_SHUTOFF_NOTICE = "has_utility_shutoff_notice"
    HAS_EVICTION_NOTICE = "has_eviction_notice"
    FILED_TAXES_LAST_YEAR = "filed_taxes_last_year"


class CategoricalProfileField(StrEnum):
    """Profile fields a :class:`~app.schemas.eligibility.CategoricalCondition`
    may test.

    Restricting the field name to a closed enum means a malformed rule fails at
    load time instead of silently evaluating to ``UNKNOWN`` forever.
    """

    STATE = "state"
    COUNTY = "county"
    STUDENT_STATUS = "student_status"
    EMPLOYMENT_STATUS = "employment_status"
    HOUSING_SITUATION = "housing_situation"


#: Which program categories are worth retrieving for a stated need. Used by the
#: retrieval layer as a pre-filter, never as an eligibility judgement.
NEED_TO_PROGRAM_CATEGORIES: Final[dict[NeedCategory, tuple[ProgramCategory, ...]]] = {
    NeedCategory.FOOD: (ProgramCategory.FOOD, ProgramCategory.COMMUNITY_SERVICES),
    NeedCategory.HEALTHCARE: (ProgramCategory.HEALTHCARE,),
    NeedCategory.UTILITIES: (ProgramCategory.UTILITIES,),
    NeedCategory.HOUSING: (ProgramCategory.HOUSING, ProgramCategory.COMMUNITY_SERVICES),
    NeedCategory.CHILDCARE: (ProgramCategory.CHILDCARE,),
    NeedCategory.TRANSPORTATION: (ProgramCategory.COMMUNITY_SERVICES,),
    NeedCategory.EDUCATION_COSTS: (
        ProgramCategory.UNIVERSITY_EMERGENCY_AID,
        ProgramCategory.COMMUNITY_SERVICES,
    ),
    NeedCategory.INCOME_SUPPORT: (
        ProgramCategory.CASH_TAX_CREDITS,
        ProgramCategory.COMMUNITY_SERVICES,
    ),
    NeedCategory.INTERNET_PHONE: (ProgramCategory.UTILITIES,),
    NeedCategory.LEGAL_HELP: (ProgramCategory.COMMUNITY_SERVICES,),
    NeedCategory.MENTAL_HEALTH: (ProgramCategory.HEALTHCARE, ProgramCategory.COMMUNITY_SERVICES),
    NeedCategory.EMERGENCY_EXPENSE: (
        ProgramCategory.UNIVERSITY_EMERGENCY_AID,
        ProgramCategory.COMMUNITY_SERVICES,
    ),
    NeedCategory.OTHER: (),
    NeedCategory.UNKNOWN: (),
}
