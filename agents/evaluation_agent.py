from smolagents import HfApiModel, CodeAgent, LiteLLMModel

from config.model import LargeLanguageModel
from tools.semantic import calculate_text_similarity
from agents.agent import get_smolagents_agent


class EvaluationAgent(CodeAgent):
    model: HfApiModel | LiteLLMModel

    def __init__(
        self,
        model: LargeLanguageModel = LargeLanguageModel.PHI_4,
    ):
        self.model = get_smolagents_agent(model)

        super().__init__(
            tools=[calculate_text_similarity],
            model=self.model,
            max_steps=5,
            verbosity_level=2,
            name="evaluation_agent",
            description="Evaluates the original and refined queries ",
        )
        # TODO: Based off logs, this prompt doesn't seem to be passed in
        self.system_prompt = """
        Role: You are an evaluation agent responsible for assessing the quality of refined queries for semantic search. Your goal is to determine which refined query is most effective for retrieving relevant and high-quality information.

        For each refined query, assess the following:
        1. Semantic Similarity: Does the refined query preserve the intent of the original query while improving clarity?
        2. Retrieval Effectiveness: Would this query improve search results by making it more specific, structured, or relevant?
        3. Query Expansion Quality: If expanded, does it add meaningful context without introducing irrelevant details?
        4. Query Decomposition: If split into subqueries, do they comprehensively cover the original intent?
        5. Embedding Distance: Compare the embeddings of the original query and the refined query to ensure a meaningful difference.

        Task Instructions:
        1. Compare each refined query against the original based on the evaluation criteria.
        2. Assign a score (0-10) to each refined query based on how well it improves search effectiveness.
        3. Select the best refined query and justify why it performs better.
        """
