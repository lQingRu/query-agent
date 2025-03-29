from datetime import datetime
from agents.state import EvaluationResultQuestion
from langgraph.types import Command
from langgraph.graph import END


def evaluation_orchestrator(state: EvaluationResultQuestion):
    print(f"Stage: evaluation_orchestrator, Time: {datetime.now()}")
    print(state)

    _overall_state = {
        "original_question": state.question,
        "evaluation_results": {
            "question_structure_eval": state.question_structure_eval,
            "abbreviations_eval": state.abbreviations_eval,
            "domain_specific_term_eval": state.domain_specific_term_eval,
            "keywords_eval": state.keywords_eval,
        },
        "refined_questions": state.refined_questions,
    }
    if (
        state.question_structure_eval.score <= 4
        or len(state.abbreviations_eval.abbreviations) > 0
        or len(state.domain_specific_term_eval.domain_terms) > 0
    ):
        return Command(update=_overall_state, goto="human_selection_node")
    else:
        print("Question looks good!")
        return Command(update=_overall_state, goto=END)
