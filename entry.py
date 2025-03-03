from agents.planning_agent import PlanningAgent


if __name__ == "__main__":
    planning_agent = PlanningAgent()
    planning_agent.visualize()
    planning_agent.run("Any recent news on Ukraine and US?")
