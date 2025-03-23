from agents.evaluation.abbreviations import abbreviations_eval
from agents.evaluation.evaluator_orchestrator import evaluation_orchestrator
from agents.evaluation.question_structure import question_structure_eval_agent
from agents.evaluation.domain_specific_terms import domain_specific_terms_eval_agent

from agents.evaluation.keywords import keywords_eval_agent

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.refinement.refine_question import refine_question_eval
from agents.state import InitialState

builder = StateGraph(InitialState)
builder.add_node(question_structure_eval_agent)
builder.add_node(abbreviations_eval)
builder.add_node(keywords_eval_agent)
builder.add_node(domain_specific_terms_eval_agent)
builder.add_node(refine_question_eval)
builder.add_node(evaluation_orchestrator)

builder.add_edge(START, "question_structure_eval_agent")
builder.add_edge(START, "abbreviations_eval")
builder.add_edge(START, "keywords_eval_agent")
builder.add_edge(START, "domain_specific_terms_eval_agent")

builder.add_edge(
    [
        "question_structure_eval_agent",
        "abbreviations_eval",
        "keywords_eval_agent",
        "domain_specific_terms_eval_agent",
    ],
    "evaluation_orchestrator",
)
builder.add_edge("refine_question_eval", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


def initialize_state(original_question: str) -> InitialState:
    """Initializes the State with default values."""
    return {"question": original_question}


state = initialize_state("What is the impact of LoRA on transformer efficiency?")

config = {"configurable": {"thread_id": "1"}}

for s in graph.stream(state, config, stream_mode="values"):
    print("=====Current Stage====")
    print(s)
