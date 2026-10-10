"""Assembling a validated :class:`~app.schemas.profile.UserProfile`.

``docs/INTERFACES.md`` assigns this policy to
``app.services.profile_assembly.assemble_profile``, which has not landed yet.
The graph cannot run without it, so this module implements exactly that
signature and :class:`~app.workflows.deps.WorkflowDeps` takes it as an injected
callable - when the services module appears, the swap is one argument and
nothing here needs to change. See the handoff note in the task report.

The precedence, highest first:

    clarification answers > prefilled > base profile > extracted

Extraction loses to everything because it is the only source that guessed. A
clarification answer wins because the user was asked the question directly, and
a *skipped* answer is a real answer: it records the field as unknown so the
clarification step stops asking.

Nothing here infers. An absent field stays absent, which the rule engine reads
as ``UNKNOWN`` and reports as ``MORE_INFO_NEEDED`` - never as a reason to
discourage somebody from applying.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from pydantic import ValidationError

from app.core.logging import get_logger
from app.schemas import (
    PROFILE_FIELD_NAMES,
    ClarificationAnswer,
    EmploymentStatus,
    ExtractedProfile,
    HousingSituation,
    IncomePeriod,
    NeedCategory,
    ProfileDraft,
    ProfileFlag,
    RangeValue,
    StudentStatus,
    UserProfile,
)

__all__ = ["INCOME_FIELDS", "answer_to_fields", "assemble_profile"]

_log = get_logger(__name__)

#: The three columns an ``income`` answer resolves. Treated as one unit: an
#: amount without a period is a validation error by design, so they are always
#: written or dropped together.
INCOME_FIELDS: Final[tuple[str, ...]] = (
    "income_min_cents",
    "income_max_cents",
    "income_period",
)

_CENTS_PER_DOLLAR: Final[int] = 100
_MAX_DROP_PASSES: Final[int] = 4

_ENUM_BY_FIELD: Final[dict[str, type[StudentStatus | EmploymentStatus | HousingSituation]]] = {
    "student_status": StudentStatus,
    "employment_status": EmploymentStatus,
    "housing_situation": HousingSituation,
}


def _dollars_to_cents(value: float) -> int:
    """Dollars to whole cents, rounded outward is the caller's job.

    A bare amount is floored for a minimum and ceilinged for a maximum by
    :func:`_income_from_range`; this helper only converts.
    """
    return round(value * _CENTS_PER_DOLLAR)


def _income_from_range(value: RangeValue) -> dict[str, Any]:
    """A monthly income band, in cents, from a ``range``-kind answer.

    Rounded outward - minimum floored, maximum ceilinged - for the same reason
    :meth:`UserProfile.income_monthly_range_cents` does it: widening a range can
    only turn a ``TRUE`` into an ``UNKNOWN`` (we ask), never into a ``FALSE``
    (we wrongly discourage someone).
    """
    fields: dict[str, Any] = {"income_period": IncomePeriod.MONTHLY}
    if value.min is not None:
        fields["income_min_cents"] = max(0, math.floor(value.min * _CENTS_PER_DOLLAR))
    if value.max is not None:
        fields["income_max_cents"] = max(0, math.ceil(value.max * _CENTS_PER_DOLLAR))
    return fields


def _bool_of(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        folded = value.strip().casefold()
        if folded in {"true", "yes", "y", "1"}:
            return True
        if folded in {"false", "no", "n", "0"}:
            return False
    return None


def answer_to_fields(answer: ClarificationAnswer) -> dict[str, Any]:
    """Project one clarification answer onto ``UserProfile`` field names.

    Returns an empty mapping for a skipped answer, for a value whose shape does
    not fit the field, and for anything outside
    :data:`~app.schemas.profile.PROFILE_FIELD_NAMES`. Refusing a bad value is
    always right here: the alternative is writing a guess into the one object
    the rule engine trusts.
    """
    field = answer.field
    value = answer.value
    if answer.skipped or value is None or field not in PROFILE_FIELD_NAMES:
        return {}

    if field.startswith("flags."):
        flag_value = _bool_of(value)
        if flag_value is None:
            return {}
        return {"flags": {ProfileFlag(field.removeprefix("flags.")): flag_value}}

    if field in {"income", *INCOME_FIELDS}:
        if isinstance(value, RangeValue):
            return _income_from_range(value)
        if isinstance(value, Mapping):
            return _income_from_range(RangeValue.model_validate(value))
        if isinstance(value, int | float) and not isinstance(value, bool):
            cents = _dollars_to_cents(float(value))
            return {
                "income_min_cents": cents,
                "income_max_cents": cents,
                "income_period": IncomePeriod.MONTHLY,
            }
        return {}

    if field in {"household_size", "age"}:
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            return {}
        try:
            return {field: int(float(value))}
        except (TypeError, ValueError):
            return {}

    if field in _ENUM_BY_FIELD:
        if not isinstance(value, str):
            return {}
        try:
            return {field: _ENUM_BY_FIELD[field](value)}
        except ValueError:
            return {}

    if field == "needs":
        if not isinstance(value, list):
            return {}
        needs: list[NeedCategory] = []
        for item in value:
            try:
                needs.append(NeedCategory(item))
            except ValueError:
                continue
        return {"needs": needs} if needs else {}

    if field in {"state", "county"}:
        if not isinstance(value, str) or not value.strip():
            return {}
        cleaned = value.strip()
        return {field: cleaned.upper() if field == "state" else cleaned}

    return {}


def _apply(fields: dict[str, Any], update: Mapping[str, Any]) -> list[str]:
    """Apply one precedence layer in place and report the names it set.

    ``flags`` is merged key by key rather than replaced, because two layers can
    legitimately know different flags, and a later layer that knows one flag
    must not erase a flag the user stated earlier.
    """
    touched: list[str] = []
    for name, value in update.items():
        if value is None:
            continue
        if name == "flags":
            merged = dict(fields.get("flags") or {})
            for flag, flag_value in dict(value).items():
                key = ProfileFlag(flag)
                merged[key] = bool(flag_value)
                touched.append(f"flags.{key.value}")
            fields["flags"] = merged
            continue
        fields[name] = value
        touched.append(name)
    return touched


def _flag_items(profile: UserProfile | None) -> dict[ProfileFlag, bool]:
    return dict(profile.flags) if profile is not None else {}


def _known_names(fields: Mapping[str, Any]) -> set[str]:
    """Every field name the assembled mapping actually determines."""
    known = {name for name, value in fields.items() if value is not None and name != "flags"}
    known |= {f"flags.{flag.value}" for flag in (fields.get("flags") or {})}
    if any(name in fields for name in INCOME_FIELDS) and fields.get("income_period") is not None:
        known.add("income")
    return known


def _validate_with_drops(fields: dict[str, Any]) -> tuple[UserProfile, list[str]]:
    """Validate, dropping the fields pydantic objects to, until it passes.

    Extraction and a stale prefill can disagree - a household composition that
    sums past the household size, an income floor above its ceiling. Dropping
    the offending field and keeping the rest is strictly better than failing
    the whole run, and every drop is returned so the caller can record it as a
    limitation rather than hide it.
    """
    dropped: list[str] = []
    working = dict(fields)
    for _ in range(_MAX_DROP_PASSES):
        try:
            return UserProfile(**working), dropped
        except ValidationError as exc:
            removable: list[str] = []
            for error in exc.errors():
                location = [str(part) for part in error["loc"]]
                candidates = location[:1] or ["household_composition"]
                for name in candidates:
                    if name in working:
                        removable.append(name)
            # A model-level validator reports no field at all; the two rules
            # that can fire name the pair they object to, so drop both halves.
            if not removable:
                removable = [
                    name for name in ("household_composition", *INCOME_FIELDS) if name in working
                ]
            if not removable:
                raise
            for name in dict.fromkeys(removable):
                working.pop(name, None)
                dropped.append(name)
    return UserProfile(), [*dropped, *working]


def assemble_profile(
    extracted: ExtractedProfile | None,
    *,
    base: UserProfile | None = None,
    prefilled: ProfileDraft | None = None,
    answers: Sequence[ClarificationAnswer] = (),
    min_confidence: float = 0.5,
) -> tuple[UserProfile, list[str]]:
    """Merge every source of profile knowledge into one validated profile.

    :returns: ``(profile, changed_fields)``. ``changed_fields`` names the fields
        whose value differs from ``base``, which is what drives selective
        re-evaluation: a household-size edit must not re-evaluate, re-explain
        and re-bill a program whose rules never read household size.
    """
    fields: dict[str, Any] = {}
    unknown: dict[str, None] = {}

    if extracted is not None:
        _apply(fields, extracted.to_profile_fields(min_confidence=min_confidence))
        # ``ExtractedProfile.unknown_fields`` is deliberately *not* merged into
        # ``UserProfile.unknown_fields``. They sound alike and mean opposite
        # things: the model's list is "the narrative did not say", which is
        # exactly the case a clarification question exists to resolve, while
        # the profile's list is "the user was asked and declined", which stops
        # us asking again. Conflating them would silently suppress every
        # question the product needs to ask.

    if base is not None:
        base_fields = base.model_dump(exclude_none=True, exclude={"unknown_fields", "extras"})
        base_fields["flags"] = _flag_items(base)
        if not base.needs:
            base_fields.pop("needs", None)
        if base.household_composition.is_empty():
            base_fields.pop("household_composition", None)
        _apply(fields, base_fields)
        for name in base.unknown_fields:
            unknown.setdefault(name, None)
        if base.extras:
            fields["extras"] = dict(base.extras)

    if prefilled is not None:
        _apply(fields, prefilled.set_fields())

    changed_by_answer: list[str] = []
    for answer in answers:
        if answer.skipped:
            if answer.field in PROFILE_FIELD_NAMES:
                unknown.setdefault(answer.field, None)
            continue
        projected = answer_to_fields(answer)
        if not projected:
            continue
        changed_by_answer.extend(_apply(fields, projected))
        if answer.field in {"income", *INCOME_FIELDS}:
            changed_by_answer.append("income")

    known = _known_names(fields)
    fields["unknown_fields"] = [name for name in unknown if name not in known]

    profile, dropped = _validate_with_drops(fields)
    if dropped:
        # Names only. A dropped field's *value* is personal content.
        _log.info("workflow.profile.conflicting_fields", fields=sorted(set(dropped)))

    changed = _changed_fields(base, profile)
    for name in changed_by_answer:
        if name not in changed and name not in dropped:
            changed.append(name)
    return profile, changed


def _changed_fields(base: UserProfile | None, profile: UserProfile) -> list[str]:
    """Which profile fields ``profile`` states differently from ``base``."""
    if base is None:
        initial = [
            name
            for name in profile.model_dump(exclude_none=True, exclude={"unknown_fields", "extras"})
            if name in PROFILE_FIELD_NAMES and not profile.is_unknown(name)
        ]
        initial.extend(f"flags.{flag.value}" for flag in profile.flags)
        if profile.has_income_information():
            initial.append("income")
        return list(dict.fromkeys(initial))

    changed: list[str] = []
    for name in sorted(PROFILE_FIELD_NAMES):
        if name.startswith("flags."):
            flag = ProfileFlag(name.removeprefix("flags."))
            if base.flag(flag) != profile.flag(flag):
                changed.append(name)
            continue
        if name == "income":
            if base.income_monthly_range_cents() != profile.income_monthly_range_cents():
                changed.append(name)
            continue
        if getattr(base, name, None) != getattr(profile, name, None):
            changed.append(name)
    return changed
