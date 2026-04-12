"""
agent.py
--------
Main Agent class. One instance per A2A context_id (one negotiation session).

The goal for the flow per incoming message:
  1. Parse the green agent's observation.
  2. Update SessionState.
  3. LLM call with the circle-5 style prompt.
  4. Parse LLM response; validate against M1-M5.
  5. On violation with auto-fix -> use the fix; on violation without fix -> retry LLM (once).
  6. On total failure -> emit a safe heuristic action that provably satisfies M1-M5.
  7. Record my chosen offer in state for next turn's M1 check.
"""

from __future__ import annotations

import logging

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message

from core.llm import LLM
from core.parser import parse_incoming, parse_llm_action, format_action

from .state import SessionState
from .negotiation import (
    ValidationResult,
    clamp_offer,
    is_offer_feasible,
    my_value_of_my_offer,
    my_value_of_their_offer,
    target_value_for_round,
    validate_action,
    value_of,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt, build_retry_prompt

logger = logging.getLogger(__name__)

MAX_LLM_RETRIES = 1  
# one retry, then fall back to heuristic

class Agent:
    def __init__(self):
        self.state = SessionState()
        self.llm = LLM()

    async def run(self, message: Message, updater: TaskUpdater) -> str:
        raw_text = get_message_text(message)
        logger.info("Incoming: %s", raw_text[:300])

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Analyzing offer..."),
        )

        obs = parse_incoming(raw_text)
        if obs:
            self.state.update_from_observation(obs)

        # If we have no valuations, we can't reason — just return a safe
        # heuristic so the game continues.
        if not self.state.my_valuations:
            logger.warning("No valuations available; returning safe WALK.")
            return format_action({"action": "WALK"})

        action = await self._decide()
        if action.get("action") == "COUNTEROFFER":
            self.state.record_my_offer(action.get("offer", []))
        return format_action(action)

    # Decision pipeline
    async def _decide(self) -> dict:
        # 1) First LLM attempt
        action = await self._ask_llm(retry_violations=None)

        if action is not None:
            validated = self._validate_and_fix(action)
            if validated is not None:
                return validated

        # 2) One retry with explicit violation feedback
        violations = self._last_violations or ["Previous response was malformed."]
        action = await self._ask_llm(retry_violations=violations)
        if action is not None:
            validated = self._validate_and_fix(action)
            if validated is not None:
                return validated

        # 3) Heuristic fallback — guaranteed M1-M5-safe
        logger.warning("LLM failed after retries; using heuristic fallback.")
        return self._heuristic_action()

    _last_violations: list[str] | None = None

    async def _ask_llm(self, retry_violations: list[str] | None) -> dict | None:
        if retry_violations:
            user = build_retry_prompt(self.state, retry_violations)
        else:
            user = build_user_prompt(self.state)
        try:
            raw = await self.llm.complete(SYSTEM_PROMPT, user)
        except Exception as exc:
            logger.exception("LLM call failed: %s", exc)
            return None
        logger.info("LLM raw: %s", raw[:200])
        return parse_llm_action(raw)

    def _validate_and_fix(self, action: dict) -> dict | None:
        """
        Run validator. If clean -> return as-is.
        If violations with auto-fix -> apply fix and re-validate; return if clean.
        If violations with no fix -> stash violations for retry prompt, return None.
        """
        result = validate_action(
            action,
            quantities=self.state.quantities,
            my_valuations=self.state.my_valuations,
            batna=self.state.batna,
            my_previous_offer=self.state.my_previous_offer,
            their_last_offer=self.state.their_last_offer,
            is_last_round=self.state.is_last_round,
        )
        if result.ok:
            self._last_violations = None
            return action

        logger.warning("Action violated: %s", result.violations)
        self._last_violations = list(result.violations)

        if result.fixed_action is not None:
            # Re-check the fix (fix might itself need re-clamping etc.)
            refix = validate_action(
                result.fixed_action,
                quantities=self.state.quantities,
                my_valuations=self.state.my_valuations,
                batna=self.state.batna,
                my_previous_offer=self.state.my_previous_offer,
                their_last_offer=self.state.their_last_offer,
                is_last_round=self.state.is_last_round,
            )
            if refix.ok:
                logger.info("Applied auto-fix: %s", result.fixed_action)
                return result.fixed_action

        return None

    # Heuristic fallback; always M1-M5 safe
    def _heuristic_action(self) -> dict:
        s = self.state
        # If they're offering something strictly better than BATNA, ACCEPT
        if s.their_last_offer is not None:
            their_val = my_value_of_their_offer(
                s.their_last_offer, s.quantities, s.my_valuations
            )
            if their_val >= s.batna and s.is_last_round:
                return {"action": "ACCEPT"}
            if their_val > s.batna * 1.2:
                return {"action": "ACCEPT"}

        # Otherwise craft a counteroffer that targets the concession schedule
        # and is guaranteed to dominate our previous offer
        if not s.my_valuations or not s.quantities:
            return {"action": "WALK"}

        max_val = value_of(s.quantities, s.my_valuations)
        target = target_value_for_round(
            s.current_round, s.max_rounds, max_val, s.batna
        )

        # Greedy: take items in order of my per-item value until value >= target
        order = sorted(range(len(s.my_valuations)), key=lambda i: -s.my_valuations[i])
        offer = [0] * len(s.quantities)
        for i in order:
            while offer[i] < s.quantities[i] and value_of(offer, s.my_valuations) < target:
                offer[i] += 1
            if value_of(offer, s.my_valuations) >= target:
                break

        # Don't be extreme (M3): if we took everything, drop one unit of the
        # item we value least (but still keep above BATNA and previous offer)
        total_taken = sum(offer)
        total_available = sum(s.quantities)
        if total_taken == total_available and total_available > 1:
            low = sorted(range(len(s.my_valuations)), key=lambda i: s.my_valuations[i])
            for i in low:
                if offer[i] > 0:
                    trial = list(offer)
                    trial[i] -= 1
                    min_floor = max(s.batna, 0.0)
                    if s.my_previous_offer is not None:
                        prev = my_value_of_my_offer(s.my_previous_offer, s.my_valuations)
                        min_floor = max(min_floor, prev)
                    if value_of(trial, s.my_valuations) >= min_floor:
                        offer = trial
                        break

        # Final safety: enforce >= previous offer (M1) and >= BATNA (M2)
        if s.my_previous_offer is not None:
            prev_val = my_value_of_my_offer(s.my_previous_offer, s.my_valuations)
            if value_of(offer, s.my_valuations) < prev_val:
                offer = list(s.my_previous_offer)
        if value_of(offer, s.my_valuations) < s.batna:
            # Take max bundle
            offer = list(s.quantities)

        offer = clamp_offer(offer, s.quantities)
        if not is_offer_feasible(offer, s.quantities):
            return {"action": "WALK"}
        return {"action": "COUNTEROFFER", "offer": offer}
