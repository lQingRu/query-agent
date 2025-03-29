import logging
from agents.state import HumanSelectionChanges, OverallState
from langgraph.types import interrupt


def human_selection_node(state: OverallState) -> HumanSelectionChanges:
    selection = interrupt(
        {
            "task": "Select the changes to be corrected to your question",
            "llm_generated": state.model_dump_json(),
        }
    )

    try:
        _selection_object = HumanSelectionChanges.model_validate(selection)

    except Exception:
        logging.error("Wrong input state!")

    return _selection_object.model_dump_json()
