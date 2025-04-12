from agents.evaluation.abbreviations import abbreviations_eval
from agents.evaluation.evaluator_orchestrator import evaluation_orchestrator
from agents.evaluation.question_structure import question_structure_eval_agent
from agents.evaluation.domain_specific_terms import domain_specific_terms_eval_agent

from agents.evaluation.keywords import keywords_eval_agent

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.refinement.human_selection import human_selection_node
from agents.refinement.refine_question import refine_question_eval
from agents.refinement.refine_question_structure import refine_question_structure_node
from agents.refinement.refine_question_with_human_input import (
    refine_question_with_human_input_node,
)
from agents.state import InitialState


def run_langgraph(state: InitialState):

    builder = StateGraph(InitialState)
    builder.add_node(question_structure_eval_agent)
    builder.add_node(abbreviations_eval)
    builder.add_node(keywords_eval_agent)
    builder.add_node(domain_specific_terms_eval_agent)
    builder.add_node(refine_question_eval)
    builder.add_node(evaluation_orchestrator)

    # With human inputs
    builder.add_node(human_selection_node)
    builder.add_node(refine_question_with_human_input_node)
    builder.add_node(refine_question_structure_node)

    builder.add_edge(START, "question_structure_eval_agent")
    builder.add_edge(START, "abbreviations_eval")
    builder.add_edge(START, "keywords_eval_agent")
    builder.add_edge(START, "domain_specific_terms_eval_agent")
    builder.add_edge("question_structure_eval_agent", "refine_question_structure_node")

    builder.add_edge(
        [
            "refine_question_structure_node",
            "abbreviations_eval",
            "domain_specific_terms_eval_agent",
            "keywords_eval_agent",
        ],
        "evaluation_orchestrator",
    )

    builder.add_edge("human_selection_node", "refine_question_with_human_input_node")
    builder.add_edge("refine_question_with_human_input_node", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    # Stream token by token (take note the yield)
    config = {"configurable": {"thread_id": "1"}}
    for message_chunk, metadata in graph.stream(state, config, stream_mode="messages"):
        if message_chunk.content:
            yield f"{message_chunk.content}\n"


# def initialize_state(original_question: str) -> InitialState:
#     """Initializes the State with default values."""
#     return {"question": original_question}


# state = initialize_state("What is the impact of LoRA on transformer efficiency?")
# config = {"configurable": {"thread_id": "1"}}

# # Stream token by token
# for message_chunk, metadata in graph.stream(state, config, stream_mode="messages"):
#     if message_chunk.content:
#         print(message_chunk.content, flush=True)
