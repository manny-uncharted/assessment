import json
import os
from crewai import Agent, Task, Crew
from langchain.llms import Ollama
# Assuming BrowserTools is correctly imported from your tools module
from tools import BrowserTools, serper_tool


class EducationalContentCrew:
    def __init__(self, subject, llm_model="openhermes"):
        self.llm = Ollama(model=llm_model)
        self.agents = []
        self.tasks = []
        self.subject = subject
        self.__create_agents_and_tasks()

    def __create_agents_and_tasks(self):
        # Define the Researcher Agent
        researcher = Agent(
            role='Researcher',
            goal="Develop ideas for teaching someone new to Biology and ecology.",
            backstory=f"You are a researcher who develops ideas for teaching someone new to {self.subject}.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools = [serper_tool]
        )

        # Define the Research Task
        task_research = Task(
            description=f"Develop a topic for teaching someone new to {self.subject}.",
            agent=researcher
        )

        # Define the Writer Agent
        writer = Agent(
            role='Writer',
            goal="Use the Researcher’s ideas to write a piece of text to explain the topic.",
            backstory="You are a writer who uses the researcher's ideas to write a detailed explanation of the topic.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools = [serper_tool]
        )

        # Define the Writing Task
        task_write = Task(
            description="Write a detailed explanation of the topic using the research ideas.",
            agent=writer
        )

        # Define the Examiner Agent
        examiner = Agent(
            role='Examiner',
            goal="Craft 2-3 test questions to evaluate understanding of the created text, along with the correct answers.",
            backstory="You are an examiner who crafts test questions to evaluate understanding of the text.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools = [serper_tool]
        )

        # Define the Examination Task
        task_examine = Task(
            description="Craft 2-3 test questions and answers to evaluate understanding of the created text.",
            agent=examiner
        )

        self.agents.extend([researcher, writer, examiner])
        self.tasks.extend([task_research, task_write, task_examine])

    def run(self):
        # Initialize the Crew with the defined agents and tasks
        crew = Crew(
            agents=self.agents,
            tasks=self.tasks,
            verbose=2
        )

        # Execute the workflow
        result = crew.kickoff()
        return result

def main():
    print("## Educational Content Generation Workflow")
    subject = "biology"
    content_crew = EducationalContentCrew(subject=subject)
    result = content_crew.run()

    print("\n## Generated Content and Evaluation")
    print(result)

if __name__ == "__main__":
    main()
