"""
executor.py
-----------
A2A executor: one Agent instance per context_id.

Near-verbatim adaptation of the Sprint 1 BWIM executor — the plumbing
pattern for A2A is identical across benchmarks.
"""

from __future__ import annotations

import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from agent.agent import Agent

logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    def __init__(self):
        self._agents: dict[str, Agent] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already terminal: {task.status.state}"
                )
            )

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self._agents.get(context_id)
        if agent is None:
            logger.info("New Agent for context_id=%s", context_id)
            agent = Agent()
            self._agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()

        try:
            response = await agent.run(msg, updater)
            await updater.complete(
                new_agent_text_message(response) if response else None
            )
        except Exception as exc:
            logger.exception("Agent error in context %s: %s", context_id, exc)
            await updater.failed(
                new_agent_text_message(
                    f"Agent error: {exc}",
                    context_id=context_id,
                    task_id=task.id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
