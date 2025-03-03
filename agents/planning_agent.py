from smolagents import HfApiModel, CodeAgent, LiteLLMModel

from agents.agent import get_smolagents_agent
from agents.evaluation_agent import EvaluationAgent
from config.model import (
    LargeLanguageModel,
)


class PlanningAgent(CodeAgent):
    model: HfApiModel | LiteLLMModel
    task: str

    def __init__(
        self,
        model: LargeLanguageModel = LargeLanguageModel.PHI_4,
        task: str = None,
    ):
        self.model = get_smolagents_agent(model)
        evaluation_agent = EvaluationAgent()

        super().__init__(
            tools=[],
            model=self.model,
            managed_agents=[evaluation_agent],
            max_steps=5,
            verbosity_level=2,
            name="planning_agent",
            description="Refines the steps required to refine user's search query ",
        )
        self.task = (
            task
            or """
        Role: You are a query planning agent specializing in crafting effective natural language queries for semantic search.

        Key Considerations for Refinement:
        1. Question Structure: Ensure clarity and specificity.
        2. Abbreviations: Expand to improve understanding.
        3. Domain-Specific Terminology: Use precise terms relevant to the topic.
        4. Keywords: Emphasize essential terms for better retrieval.
        5. Open vs. Closed-Ended: Adjust based on the retrieval goal.

        Objective: Your goal is to come up with a few refined queries for optimal semantic search performance.
        Once you have these refined queries, you should pass these refined queries and the original query to the evaluation agent to evaluate.
        """
        )

    def run(self, question: str):
        prompt = f"""
        {self.task}
        Question: {question}
        """
        return super().run(prompt)
