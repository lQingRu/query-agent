from agents.state import HumanSelectionChanges
from config.model import LargeLanguageModel, llm
from langchain_core.prompts import PromptTemplate


def refine_question_with_human_input_node(state: HumanSelectionChanges):
    PROMPT_TEMPLATE = """
    You are tasked with refining a user's question while maintaining its original intent and improving clarity. 
    
    # User's Question
    {question}
    
    # Instructions
    {instructions}
    
    Refine the user's question based on the instructions given by returning only a single refined question.
    """

    instructions = ""

    if state.abbreviations:
        instructions += """
        - Expand any abbreviations provided by the user, placing the expanded form in the question followed by the abbreviation in brackets.
        """

    if state.domain_specific_terms:
        instructions += """
        - Incorporate the domain-specific terms into the question, ensuring natural integration without overloading the question with more than one concept.
        """

    if state.abbreviations and len(state.abbreviations) > 0:
        abbreviation_section = "\n## Abbreviations to Expand\n" + "\n".join(
            f"- {abbr.abbreviation}: {abbr.expanded}" for abbr in state.abbreviations
        )
        instructions += abbreviation_section

    if state.domain_specific_terms and len(state.domain_specific_terms) > 0:
        domain_terms_section = "\n## Domain-Specific Terms to Integrate\n" + ", ".join(
            state.domain_specific_terms
        )
        instructions += domain_terms_section

    model = llm(model=LargeLanguageModel.PHI_4)
    instructions = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["question", "instructions"],
    )
    chain = instructions | model

    response = chain.invoke(
        {
            "question": state.question,
            "instructions": instructions,
        }
    )
    print("[results] refine_question_with_human_input: ")
    print(response)
