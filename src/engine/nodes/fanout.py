"""FanOut / FanIn nodes: parallel group step coordination.

For parallel strategy, steps with the same "group" tag are executed
consecutively without quality gate checks between them. After all group
steps complete, the batch passes through quality gate together.

This is a simplified but practical approach — true LangGraph Send-based
parallelism can be added later for I/O-bound tasks.
"""
import logging

from src.engine.state import WorkflowState

logger = logging.getLogger(__name__)


class FanOutNode:
    """Identify the current parallel group and set up batch execution.

    Scans the plan for steps sharing the same "group" tag, marks them
    for sequential batch execution, and sets batch tracking state.
    """

    def __call__(self, state: WorkflowState) -> dict:
        """Determine the current parallel batch to execute.

        Returns state update with:
        - current_step: first step in the batch
        - batch_steps: list of all step indices in this batch
        - batch_position: 0 (starting at first step in batch)
        """
        plan = state.get("plan", [])
        results = state.get("results", {})

        if not plan:
            return {"next_action": "finish"}

        # Find the first unexecuted step and its group
        first_step = None
        first_group = None

        for i, step in enumerate(plan):
            if str(i) not in results:
                first_step = i
                first_group = step.get("group", None)
                break

        if first_step is None:
            return {"next_action": "finish"}

        # If no group, this is a single step (not truly parallel)
        if first_group is None:
            logger.debug("FanOut: single step %d (no group)", first_step)
            return {
                "current_step": first_step,
                "next_action": "continue",
            }

        # Collect all consecutive steps in the same group
        batch_steps = []
        for i in range(first_step, len(plan)):
            if str(i) in results:
                continue
            step = plan[i]
            if step.get("group") == first_group:
                batch_steps.append(i)
            elif not step.get("group"):
                continue  # skip non-grouped steps — they come after
            else:
                break  # different group, stop here

        logger.info(
            "FanOut: parallel group '%s' → batch of %d steps: %s",
            first_group, len(batch_steps), batch_steps,
        )

        return {
            "current_step": batch_steps[0],
            "next_action": "continue",
            # Extended state for batch tracking
            "_batch_steps": batch_steps,
            "_batch_position": 0,
        }


class FanInNode:
    """Collect results from a parallel batch and advance past the group.

    After all steps in a batch complete, merge their results and set
    current_step past the entire group so the next router iteration
    picks up the step after the group.
    """

    def __call__(self, state: WorkflowState) -> dict:
        """Merge batch results and advance to the step after the group.

        All batch steps already executed — just update current_step to
        jump past the batch so the next iteration starts fresh.
        """
        plan = state.get("plan", [])
        results = state.get("results", {})

        # Find the max executed step index to skip past
        max_done = -1
        for key in results:
            try:
                step_idx = int(key)
                if step_idx > max_done:
                    max_done = step_idx
            except (ValueError, TypeError):
                pass

        next_step = max_done + 1

        # Clean up batch tracking fields
        return {
            "current_step": next_step,
            "_batch_steps": [],
            "_batch_position": 0,
        }
