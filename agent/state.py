"""
state.py
--------
Per-negotiation-session state. One SessionState per A2A context_id.

Tracks:
  - My role ("row" / "col") and fixed game parameters
  - My private valuations and BATNA
  - Sequence of (my_offer, their_offer) exchanges
  - Round counter
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionState:
    role: str = "row"
    quantities: list[int] = field(default_factory=lambda: [7, 4, 1])
    my_valuations: list[float] = field(default_factory=list)
    batna: float = 0.0
    max_rounds: int = 5
    discount: float = 0.98

    # History
    my_offers: list[list[int]] = field(default_factory=list)
    their_offers: list[list[int]] = field(default_factory=list)
    current_round: int = 0

    def update_from_observation(self, obs: dict) -> None:
        """Pull game state from a green-agent observation dict."""
        if "role" in obs:
            self.role = str(obs["role"])
        if "quantities" in obs and obs["quantities"]:
            self.quantities = [int(x) for x in obs["quantities"]]
        if "valuations" in obs and obs["valuations"]:
            self.my_valuations = [float(x) for x in obs["valuations"]]
        if "batna" in obs and obs["batna"] is not None:
            self.batna = float(obs["batna"])
        if "round" in obs and obs["round"] is not None:
            self.current_round = int(obs["round"])
        if "max_rounds" in obs and obs["max_rounds"] is not None:
            self.max_rounds = int(obs["max_rounds"])
        if "discount" in obs and obs["discount"] is not None:
            self.discount = float(obs["discount"])

        # Record their latest offer if present
        last_offer = obs.get("last_offer")
        if last_offer is not None and isinstance(last_offer, (list, tuple)):
            normalized = [int(x) for x in last_offer]
            if not self.their_offers or self.their_offers[-1] != normalized:
                self.their_offers.append(normalized)

    # Convenience accessors
    @property
    def my_previous_offer(self) -> list[int] | None:
        return self.my_offers[-1] if self.my_offers else None

    @property
    def their_last_offer(self) -> list[int] | None:
        return self.their_offers[-1] if self.their_offers else None

    @property
    def is_last_round(self) -> bool:
        return self.current_round >= self.max_rounds - 1

    def record_my_offer(self, offer: list[int]) -> None:
        self.my_offers.append(list(offer))

    def summary_for_prompt(self) -> str:
        """Compact state summary for the LLM prompt."""
        lines = [
            f"Your role: {self.role}",
            f"Round: {self.current_round + 1} of {self.max_rounds}",
            f"Quantities available: {self.quantities}",
            f"Your private valuations: {self.my_valuations}",
            f"Your BATNA (outside option): {self.batna}",
            f"Discount per round: {self.discount}",
        ]
        if self.my_offers:
            lines.append("Your prior offers (what YOU asked to keep):")
            for i, o in enumerate(self.my_offers):
                lines.append(f"  round {i}: {o}")
        if self.their_offers:
            lines.append("Opponent's prior offers (what THEY asked to keep):")
            for i, o in enumerate(self.their_offers):
                lines.append(f"  round {i}: {o}")
        return "\n".join(lines)
