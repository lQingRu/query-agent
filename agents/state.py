import operator
from typing import Annotated, Dict, List, TypedDict


class QuestionStructureEval(TypedDict):
    score: int
    reasoning: str
    proposal: str


class AbbreviationEval(TypedDict):
    class AbbreviationExpansion(TypedDict):
        abbreviation: str
        expansion: List[str]

    abbreviations: AbbreviationExpansion


class DomainSpecificTermEval(TypedDict):
    domain_terms: List[str]
    proposal: str = "Replace with synonyms or longer explanations"


class KeywordEval(TypedDict):
    class KeywordExpansion(TypedDict):
        keyword: str
        expanded_keywords: List[str]  # Fuzz & Expand

    keywords: List[KeywordExpansion]


class OpenClosedEval(TypedDict):
    reasoning: str
    proposal: str


class EvaluationResult(TypedDict):
    question_structure: Annotated[QuestionStructureEval, operator.add]
    abbreviation: Annotated[AbbreviationEval, operator.add]
    domain_specific_term: Annotated[DomainSpecificTermEval, operator.add]
    keyword: Annotated[KeywordEval, operator.add]
    open_closed: Annotated[OpenClosedEval, operator.add]


class RefinementResult(TypedDict):
    intention_score: int
    similarity_score: int
    refined_question: str


class State(TypedDict):

    class Refinement(TypedDict):
        evaluation_results: EvaluationResult
        refined_question_results: List[RefinementResult]

    original_question: str
    refinements: Dict[str, Refinement]  # <question> : RefinementResult
