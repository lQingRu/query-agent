import operator
from typing import Annotated, List

from pydantic import BaseModel


class InitialState(BaseModel):
    question: str


class QuestionStructureEval(BaseModel):
    score: int
    reasoning: str
    proposal: str


class AbbreviationEval(BaseModel):
    class AbbreviationExpansion(BaseModel):
        abbreviation: str
        expansion: List[str]

    abbreviations: AbbreviationExpansion


class DomainSpecificTermEval(BaseModel):
    domain_terms: List[str]
    proposal: str


class KeywordEval(BaseModel):
    class KeywordExpansion(BaseModel):
        keyword: str
        expanded_keywords: List[str]  # Fuzz & Expand

    keywords: List[KeywordExpansion]


class EvaluationResult(BaseModel):
    question_structure_eval: QuestionStructureEval = None
    abbreviations_eval: AbbreviationEval = None
    domain_specific_term_eval: DomainSpecificTermEval = None
    keywords_eval: KeywordEval = None


class QueryRefinementResult(BaseModel):
    intention_score: int
    similarity_score: int
    refined_question: str


class EvaluationResultQuestion(EvaluationResult):
    question: str
    refined_questions: list[str]


class OverallState(BaseModel):
    original_question: str
    evaluation_results: EvaluationResult
    refined_question_eval: Annotated[list[QueryRefinementResult], operator.add]
    refined_questions: Annotated[list[str], operator.add]


class HumanSelectionChanges(BaseModel):
    class AbbreviationExpansion(BaseModel):
        abbreviation: str
        expanded: str

    question: str
    refined_question: str
    abbreviations: List[AbbreviationExpansion]
    domain_specific_terms: List[str]


class RefineQueryStructure(BaseModel):
    question: str
    question_structure_eval: QuestionStructureEval
    refined_questions: Annotated[list[str], operator.add]
