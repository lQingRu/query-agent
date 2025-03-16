from agents.state import QuestionStructureEval, State
from langgraph.graph import END, StateGraph, START


def question_structure(state: State):
    question_structure: QuestionStructureEval = {
        "proposal": "Updates on NewWater in Singapore",
        "reasoning": "Clarify logical flow",
        "score": 3,
    }
    state["refinements"][-1]["evaluation_results"][
        "question_structure"
    ] = question_structure
    return state


def abbreviations(state: State):
    abbreviation = {
        "abbreviation": "NWS",
        "expansion": ["New Water Service"],
    }
    abbreviations = [abbreviation]

    state["refinements"][-1]["evaluation_results"]["abbreviation"] = abbreviations
    return state


def domain_specific(state: State):
    domain_results = {
        "domain_terms": ["new water"],
        "proposal": "Replace with synonyms or longer explanations",
    }
    state["refinements"][-1]["evaluation_results"][
        "domain_specific_term"
    ] = domain_results
    return state


def keyword(state: State):
    keyword_results = {
        "keyword": "new water",
        "expanded_keywords": ["new water", "xing shui"],
    }
    state["refinements"][-1]["evaluation_results"]["keyword"] = [keyword_results]
    return state


def open_closed(state: State):
    open_closed_results = {
        "proposal": "Be more specific and narrow",
        "reasoning": "Too broad, recall may be low",
    }
    state["refinements"][-1]["evaluation_results"]["open_closed"] = open_closed_results
    return state


def refine_question(state: State):
    _refined_question = "Provide updates on NewWater requirements in Singapore"
    state["refinements"][-1]["refined_question_results"].setdefault(
        _refined_question,
        {"refined_question": _refined_question},
    )
    return state


def refinement_evaluation(state: State):
    _refined_question = "Provide updates on NewWater requirements in Singapore"
    state["refinements"][-1]["refined_question_results"].setdefault(
        _refined_question,
        {
            "intention_score": 9,
            "similarity_score": 0.85,
            "refined_question": _refined_question,
        },
    )
    return state


def evaluator(state: State):
    print("Stage: Evaluating")
    current_refinement_results = state["refinements"][-1]
    if current_refinement_results["evaluation_results"]:

        eval_results = current_refinement_results["evaluation_results"]

        if current_refinement_results["refined_question_results"]:
            print(
                f"Refined Questions: {current_refinement_results['refined_question_results']}"
            )
            return "refinement_evaluate"

        print(f"Evaluation results: {eval_results['question_structure']['score']}")
        if eval_results["question_structure"]["score"] >= 5:
            print("Return to user...")
            return END
        else:
            return "refiner"


from langgraph.checkpoint.memory import MemorySaver

builder = StateGraph(State)
builder.add_node("question_structure", question_structure)
builder.add_node("abbreviations", abbreviations)
builder.add_node("keyword", keyword)
builder.add_node("open_closed", open_closed)
builder.add_node("refine_question", refine_question)
builder.add_node("refinement_evaluate", refinement_evaluation)

builder.add_edge(START, "question_structure")
builder.add_edge(START, "abbreviations")
builder.add_edge(START, "keyword")
builder.add_edge(START, "open_closed")
builder.add_edge(START, "refine_question")
builder.add_edge("question_structure", "evaluator")
builder.add_edge("abbreviations", "evaluator")
builder.add_edge("keyword", "evaluator")
builder.add_edge("open_closed", "evaluator")
builder.add_conditional_edges("evaluator", evaluator)
builder.add_edge("refinement_evaluate", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
