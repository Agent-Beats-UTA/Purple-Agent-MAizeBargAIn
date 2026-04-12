"""
negotiation.py
--------------
This is the core domain model for the MAizeBargAIn bargaining game.

The game setup (from Smithline et al. 2025 / OpenSpiel negotiation) is as follows:
  - T=3 item types with fixed quantities (default 7, 4, 1)
  - Each player has PRIVATE valuations per item, drawn uniformly from [1,100]
  - Each player has a PRIVATE BATNA (outside option / walk-away value)
  - Per-round discount factor gamma (typically 0.9 or 0.98)
  - Max R rounds (typically 3 or 5)
  - At each turn, a player can: COUNTEROFFER, ACCEPT, or WALK

Offer semantics:
  An offer [a, b, c] means "I propose that I take (a, b, c) of the three items
  and the opponent takes the rest (Q0-a, Q1-b, Q2-c)."
  (This matches the OpenSpiel convention used by the green agent.)

  The opponent's offer of [a, b, c] means "opponent wants (a, b, c), I would get (Q-a,...)"
  This symmetry MUST be carefully handled — it's a classic source of M1/M4 bugs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# Core offer / state types
@dataclass(frozen=True)
class Offer:
    """An offer is a tuple of item counts the proposer wants for themselves."""
    items: tuple[int, ...]

    def __post_init__(self):
        if any(x < 0 for x in self.items):
            raise ValueError(f"Negative item counts in offer: {self.items}")

    def as_list(self) -> list[int]:
        return list(self.items)

    def complement(self, quantities: Sequence[int]) -> "Offer":
        """What the OTHER player gets if this offer is accepted."""
        return Offer(tuple(q - x for q, x in zip(quantities, self.items)))


def value_of(offer_items: Sequence[int], valuations: Sequence[float]) -> float:
    """Dot product: my valuation of the bundle I'd get."""
    return float(sum(x * v for x, v in zip(offer_items, valuations)))


def my_value_of_their_offer(
    their_offer: Sequence[int],
    quantities: Sequence[int],
    my_valuations: Sequence[float],
) -> float:
    """
    If opponent proposes THEY take `their_offer`, I get the complement.
    Return MY value of the complement bundle.
    """
    complement = [q - x for q, x in zip(quantities, their_offer)]
    return value_of(complement, my_valuations)


def my_value_of_my_offer(
    my_offer: Sequence[int],
    my_valuations: Sequence[float],
) -> float:
    """Value to me of a bundle I'm proposing to keep."""
    return value_of(my_offer, my_valuations)


def is_offer_feasible(offer: Sequence[int], quantities: Sequence[int]) -> bool:
    """True iff every component is in [0, quantity_i]."""
    if len(offer) != len(quantities):
        return False
    return all(0 <= x <= q for x, q in zip(offer, quantities))


def clamp_offer(offer: Sequence[int], quantities: Sequence[int]) -> list[int]:
    """Snap each component into [0, q_i]."""
    return [max(0, min(int(x), q)) for x, q in zip(offer, quantities)]

# Concession schedule — sensible prior when the LLM gets wobbly
def target_value_for_round(
    round_idx: int,
    max_rounds: int,
    max_my_value: float,
    batna: float,
) -> float:
    """
    Linear concession schedule: start high (ask for near-max value),
    concede toward BATNA as rounds remaining shrinks.

    Round 0          -> ~95% of max_my_value
    Last round       -> BATNA + small epsilon
    Never below BATNA.
    """
    if max_rounds <= 1:
        return max(batna, 0.5 * max_my_value)

    # progress in [0, 1]
    progress = min(1.0, max(0.0, round_idx / max(1, max_rounds - 1)))
    top = max_my_value * 0.95
    floor = max(batna, 0.0) + 1.0
    if top < floor:
        return floor
    return top - progress * (top - floor)


def max_possible_my_value(
    quantities: Sequence[int], my_valuations: Sequence[float]
) -> float:
    """If I took everything, what would it be worth to me?"""
    return value_of(quantities, my_valuations)

@dataclass
class ValidationResult:
    """Outcome of checking an action against M1-M5."""
    ok: bool
    violations: list[str] = field(default_factory=list)
    fixed_action: dict | None = None  # suggested auto-correction, if any

    def add(self, code: str, msg: str) -> None:
        self.ok = False
        self.violations.append(f"{code}: {msg}")

def validate_action(
    action: dict,
    *,
    quantities: Sequence[int],
    my_valuations: Sequence[float],
    batna: float,
    my_previous_offer: Sequence[int] | None,
    their_last_offer: Sequence[int] | None,
    is_last_round: bool,
) -> ValidationResult:
    """
    Check a proposed action against the five classic LLM-negotiator mistakes.

    Returns ValidationResult. If violations are found AND the action can be
    salvaged (e.g., clamping an out-of-range offer), fixed_action is populated.

    Args:
        action: {"action": "COUNTEROFFER"|"ACCEPT"|"WALK", "offer": [...]?}
        quantities: item quantities available
        my_valuations: my private per-item valuations
        batna: my outside option value
        my_previous_offer: my last proposal (for M1)
        their_last_offer: opponent's most recent offer (for M4/M5)
        is_last_round: if True, ACCEPT thresholds are relaxed slightly
    """
    result = ValidationResult(ok=True)
    act = str(action.get("action", "")).upper()

    if act == "ACCEPT":
        if their_last_offer is None:
            result.add("PROTOCOL", "Cannot ACCEPT with no offer on the table.")
            return result
        my_value = my_value_of_their_offer(
            their_last_offer, quantities, my_valuations
        )
        # M4: accepting an offer worse than BATNA
        # (strict inequality; at BATNA we allow acceptance to avoid ties-at-zero)
        if my_value < batna:
            result.add(
                "M4",
                f"ACCEPTing offer worth {my_value:.1f} to me, "
                f"below BATNA {batna:.1f}.",
            )
        return result
    
    if act == "WALK":
        if their_last_offer is not None:
            my_value = my_value_of_their_offer(
                their_last_offer, quantities, my_valuations
            )
            # M5: walking from an offer BETTER than BATNA
            if my_value > batna:
                result.add(
                    "M5",
                    f"WALKing from offer worth {my_value:.1f} to me, "
                    f"above BATNA {batna:.1f}.",
                )
                # Auto-fix: accept it instead
                result.fixed_action = {"action": "ACCEPT"}
        return result

    if act == "COUNTEROFFER":
        raw_offer = action.get("offer")
        if raw_offer is None or not isinstance(raw_offer, (list, tuple)):
            result.add("PROTOCOL", f"COUNTEROFFER missing 'offer' list: {raw_offer!r}")
            return result
        if len(raw_offer) != len(quantities):
            result.add(
                "PROTOCOL",
                f"Offer length {len(raw_offer)} != item count {len(quantities)}.",
            )
            return result

        # Clamp and check feasibility
        clamped = clamp_offer(raw_offer, quantities)
        needed_fix = list(clamped) != [int(x) for x in raw_offer]

        # M3: extreme divisions — all items or zero items
        total_taken = sum(clamped)
        total_available = sum(quantities)
        if total_taken == 0 or total_taken == total_available:
            result.add(
                "M3",
                f"Extreme division: taking {total_taken}/{total_available} items.",
            )

        my_val = my_value_of_my_offer(clamped, my_valuations)

        # M2: offering something worse for me than my BATNA
        # (i.e., even if they accept, I'm below my outside option)
        if my_val < batna:
            result.add(
                "M2",
                f"Offer keeps {my_val:.1f} for me, below BATNA {batna:.1f}.",
            )
            # Auto-fix: try to take more of my highest-valued items until >= BATNA
            patched = _patch_to_meet_batna(clamped, quantities, my_valuations, batna)
            if patched is not None:
                result.fixed_action = {"action": "COUNTEROFFER", "offer": patched}
                needed_fix = True

        # M1: my new offer gives me less than my previous offer did
        if my_previous_offer is not None:
            prev_val = my_value_of_my_offer(my_previous_offer, my_valuations)
            if my_val < prev_val - 1e-6:
                result.add(
                    "M1",
                    f"New offer value {my_val:.1f} < previous {prev_val:.1f} "
                    f"(cannot walk back concessions).",
                )
                # Auto-fix: fall back to previous offer
                result.fixed_action = {
                    "action": "COUNTEROFFER",
                    "offer": list(my_previous_offer),
                }
                needed_fix = True

        # Opportunity check: M5 variant — if their last offer is already > my
        # best counteroffer's value, we should accept instead. (soft warning)
        if their_last_offer is not None:
            their_offer_my_value = my_value_of_their_offer(
                their_last_offer, quantities, my_valuations
            )
            if their_offer_my_value > my_val + 1e-6 and their_offer_my_value >= batna:
                result.add(
                    "M5*",
                    f"Their offer is worth {their_offer_my_value:.1f} to me, "
                    f"better than my {my_val:.1f} counter — should ACCEPT.",
                )
                result.fixed_action = {"action": "ACCEPT"}

        # If the offer needed clamping but no other violations, still salvage it
        if needed_fix and result.fixed_action is None:
            result.fixed_action = {"action": "COUNTEROFFER", "offer": clamped}

        return result

    result.add("PROTOCOL", f"Unknown action: {act!r}")
    return result


def _patch_to_meet_batna(
    offer: list[int],
    quantities: Sequence[int],
    my_valuations: Sequence[float],
    batna: float,
) -> list[int] | None:
    """
    Greedy repair: add units of my highest-valued items until offer >= BATNA.
    Returns None if impossible (even taking everything is below BATNA).
    """
    if value_of(quantities, my_valuations) < batna:
        return None

    # Sort item indices by my valuation, highest first
    order = sorted(range(len(my_valuations)), key=lambda i: -my_valuations[i])
    patched = list(offer)
    for i in order:
        while patched[i] < quantities[i] and value_of(patched, my_valuations) < batna:
            patched[i] += 1
        if value_of(patched, my_valuations) >= batna:
            return patched
    return None
