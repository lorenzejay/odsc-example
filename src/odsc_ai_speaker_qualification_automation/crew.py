import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import EXASearchTool


from crewai_tools import CrewaiEnterpriseTools
from pydantic import BaseModel


class EvaluationExpertiseAndReasoning(BaseModel):
    score: int
    reasoning: str


class SpeakerQualificationEvaluationScores(BaseModel):
    technical_expertise: EvaluationExpertiseAndReasoning
    speaking_experience: EvaluationExpertiseAndReasoning
    industry_impact: EvaluationExpertiseAndReasoning
    overall_recommendation: EvaluationExpertiseAndReasoning


@CrewBase
class OdscAiSpeakerQualificationAutomationCrew:
    """OdscAiSpeakerQualificationAutomation crew"""

    @agent
    def ai_speaker_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["ai_speaker_researcher"],
            tools=[EXASearchTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o",
                temperature=0.7,
            ),
        )

    @agent
    def odsc_qualification_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["odsc_qualification_specialist"],
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o",
                temperature=0.7,
            ),
        )

    @agent
    def report_messenger(self) -> Agent:
        enterprise_actions_tool = CrewaiEnterpriseTools(
            actions_list=[
                "slack_send_message",
                "slack_get_users_by_name",
            ],
        )

        return Agent(
            config=self.agents_config["report_messenger"],
            tools=[*enterprise_actions_tool],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o",
                temperature=0.7,
            ),
        )

    @task
    def research_speaker_background(self) -> Task:
        return Task(
            config=self.tasks_config["research_speaker_background"],
            markdown=False,
        )

    @task
    def evaluate_odsc_speaker_qualification(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_odsc_speaker_qualification"],
            markdown=False,
            output_pydantic=SpeakerQualificationEvaluationScores,
        )

    @task
    def send_qualification_report(self) -> Task:
        return Task(
            config=self.tasks_config["send_qualification_report"],
            markdown=False,
            guardrail="ensure that the report is sent as a slack message",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the OdscAiSpeakerQualificationAutomation crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=True,
            tracing=True,
        )
