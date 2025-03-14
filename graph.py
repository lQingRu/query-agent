from typing import TypedDict

from agents.evaluation_agent import EvaluatorOutput, evaluate_questions
from agents.refiner_agent import (
    RefinerOutput,
    refine_question,
    refine_question_with_feedback,
)

from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver


class OverallState(TypedDict):
    original_question: str
    refiner_output: RefinerOutput
    evaluator_output: EvaluatorOutput
    retries: int = 0


def reflect_refiner_node(state: OverallState):
    feedback = state["evaluator_output"].model_dump()
    response = refine_question_with_feedback(
        question=state["original_question"], feedback=feedback
    )
    return {
        "refiner_output": response,
        "retries": state["retries"] + 1,
    }


def refiner_node(state: OverallState) -> OverallState:
    response = refine_question(question=state["original_question"])
    return {"refiner_output": response}


def evaluator_node(state: OverallState) -> OverallState:
    response = evaluate_questions(
        question=state["original_question"],
        refined_questions=state["refiner_output"]["refined_questions"],
    )
    return {"evaluator_output": response}


def should_continue(state: OverallState):
    print(f"Should continue...? Messages: ")
    if state["retries"] > 2:
        return END

    if state["evaluator_output"]:
        print(state)
        try:
            if list(state["evaluator_output"]["best_refined_question"])[0].score >= 4:
                return END
        except Exception as e:
            print(f"Failed to evaluate: {e}")
            return END
    return "reflect"


builder = StateGraph(OverallState)
builder.add_node("refine", refiner_node)
builder.add_node("evaluate", evaluator_node)
builder.add_node("reflect", reflect_refiner_node)
builder.add_edge(START, "refine")
builder.add_edge("refine", "evaluate")
builder.add_conditional_edges("evaluate", should_continue)
builder.add_edge("reflect", "evaluate")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

thread = {"configurable": {"thread_id": "2"}}
results = []
for s in graph.stream(
    {
        "original_question": "Tell me the latest updates on NVDA",
        "refiner_output": None,
        "evaluator_output": None,
        "retries": 0,
    },
    thread,
):
    print(s)
    results.append(s)

print(results)
