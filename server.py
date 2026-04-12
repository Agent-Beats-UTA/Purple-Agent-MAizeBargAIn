"""
server.py
---------
A2A server entrypoint for the negotiation purple agent.
"""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from core.executor import Executor


def build_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="bargaining",
        name="Multi-round Bargaining",
        description=(
            "Plays the OpenSpiel negotiation/bargaining game used by the "
            "MAizeBargAIn meta-game assessor. Receives observations with "
            "private valuations and BATNA; responds with COUNTEROFFER, "
            "ACCEPT, or WALK. Enforces the five classic negotiation rules "
            "(M1-M5) deterministically on top of LLM reasoning."
        ),
        tags=["negotiation", "bargaining", "multi-agent", "game-theory"],
        examples=[
            '{"role":"row","round":0,"valuations":[45,72,33],"batna":85,'
            '"quantities":[7,4,1],"last_offer":null,"history":[]}',
        ],
    )
    return AgentCard(
        name="MAizeBargAIn_PurpleAgent",
        description=(
            "Purple agent for MAizeBargAIn. Combines an LLM negotiator with "
            "a deterministic M1-M5 validator and a provably-safe heuristic "
            "fallback so that every action is feasible and rule-compliant."
        ),
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9018)
    parser.add_argument("--card-url", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    debug = args.debug or os.getenv("AGENT_DEBUG", "").lower() in {"1", "true", "yes"}
    logging.basicConfig(
        level=logging.INFO if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    card_url = args.card_url
    if not card_url:
        card_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        card_url = f"http://{card_host}:{args.port}"

    handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=build_agent_card(card_url),
        http_handler=handler,
    )

    logging.getLogger(__name__).info(
        "Negotiator purple agent on %s:%d | card %s", args.host, args.port, card_url
    )
    uvicorn.run(app.build(), host=args.host, port=args.port, timeout_keep_alive=300)


if __name__ == "__main__":
    main()
