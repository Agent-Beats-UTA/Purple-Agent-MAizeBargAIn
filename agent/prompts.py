"""
prompts.py
----------
The prompt templates for the negotiation LLM call.

The strategy is to mirror the green agent's own "circle 5" prompting framework
(rules + objective + worked example + procedure + M1-M5 + numeric checks),
plus force JSON output so the response is deterministic to parse.
"""

from __future__ import annotations

from .state import SessionState
from .negotiation import (
    max_possible_my_value,
    my_value_of_their_offer,
    target_value_for_round,
)


SYSTEM_PROMPT = """You are an expert negotiator playing a multi-round bargaining game.

GAME RULES
- There are multiple item types with fixed total quantities available.
- You and your opponent each have PRIVATE per-item valuations and a PRIVATE BATNA (your outside option if no deal is reached).
- Each turn you choose exactly one action: COUNTEROFFER, ACCEPT, or WALK.
- An offer [a, b, c] means "I (the proposer) take (a, b, c); the other player takes the remainder."
- Each round applies a discount factor to all payoffs, so delay is costly.
- The game ends when someone ACCEPTs or WALKs, or after max_rounds.

OBJECTIVE
Maximize your own expected payoff: the value of items you receive (or your BATNA if you walk).
Fairness matters too — you are scored on welfare and envy-freeness, so very lopsided deals hurt you.

DECISION PROCEDURE (follow every turn)
1. Compute the value to YOU of the opponent's last offer (items you'd receive × your valuations).
2. Compare to your BATNA. If the offer is comfortably above BATNA and you're late in the game, ACCEPT.
3. Otherwise craft a COUNTEROFFER that:
   (a) Is worth MORE to you than your BATNA.
   (b) Is worth at least as much to you as your PREVIOUS offer (never walk back concessions).
   (c) Is not an extreme division (don't demand everything or offer everything).
   (d) Leaves the opponent reasonable value (estimate — they have private valuations too).
4. Only WALK if the opponent's offer is strictly worse than your BATNA and no better deal is reachable.

FIVE MISTAKES TO ABSOLUTELY AVOID
M1: Making an offer worse for YOU than your own previous offer.
M2: Making an offer worse for YOU than your BATNA.
M3: Proposing extreme divisions (all items to one side or zero items).
M4: Accepting an offer worse for YOU than your BATNA.
M5: Walking away from an offer better for YOU than your BATNA.

RESPONSE FORMAT
Respond with ONLY a JSON object on a single line, no prose, no markdown fences:
  {"action": "COUNTEROFFER", "offer": [a, b, c]}
or
  {"action": "ACCEPT"}
or
  {"action": "WALK"}

Before responding, compute your numbers silently. Double-check you have not committed M1-M5.
"""


def build_user_prompt(state: SessionState) -> str:
    """Build the user-turn prompt with concrete numbers precomputed."""
    lines: list[str] = [state.summary_for_prompt(), ""]

    max_val = max_possible_my_value(state.quantities, state.my_valuations) \
        if state.my_valuations else 0.0
    lines.append(f"If you took EVERY item, value to you = {max_val:.1f}")
    lines.append(f"Your BATNA (walk-away value) = {state.batna:.1f}")

    target = target_value_for_round(
        round_idx=state.current_round,
        max_rounds=state.max_rounds,
        max_my_value=max_val,
        batna=state.batna,
    )
    lines.append(
        f"Suggested target value for this round (concession schedule): {target:.1f}"
    )

    if state.their_last_offer is not None:
        their_val = my_value_of_their_offer(
            state.their_last_offer,
            state.quantities,
            state.my_valuations,
        )
        lines.append("")
        lines.append(
            f"Opponent's last offer: they take {state.their_last_offer}, "
            f"you would get {[q - x for q, x in zip(state.quantities, state.their_last_offer)]}"
        )
        lines.append(f"Value to YOU of accepting that offer: {their_val:.1f}")
        lines.append(f"Compare to BATNA {state.batna:.1f}: "
                     f"{'ACCEPT is safe' if their_val >= state.batna else 'BELOW BATNA — do not accept'}")

    if state.my_previous_offer is not None:
        from .negotiation import my_value_of_my_offer
        prev_val = my_value_of_my_offer(state.my_previous_offer, state.my_valuations)
        lines.append("")
        lines.append(f"Your previous offer was {state.my_previous_offer} "
                     f"(value to you: {prev_val:.1f}).")
        lines.append("Any new counteroffer must be worth AT LEAST that much to you.")

    lines.append("")
    lines.append("Now output your JSON action. No other text.")
    return "\n".join(lines)


def build_retry_prompt(state: SessionState, violations: list[str]) -> str:
    """Called when the LLM's previous response violated M1-M5 or was malformed."""
    base = build_user_prompt(state)
    return (
        base
        + "\n\nYour previous response had these problems:\n- "
        + "\n- ".join(violations)
        + "\n\nFix them and respond again with ONLY the JSON action."
    )
