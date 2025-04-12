from agents.state import InitialState
from graph import run_langgraph


if __name__ == "__main__":
    run_langgraph(InitialState(question="GE elections in sg 2025"))
