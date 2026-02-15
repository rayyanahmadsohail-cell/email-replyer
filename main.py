import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv


load_dotenv()


gemini_llm = LLM(
    model="gemini-2.5-flash", 
    api_key=os.getenv("GOOGLE_API_KEY")
)


agency_services = """
SEO Optimization Service
Target: Companies with good products/services but low website traffic
Goal: Increase organic reach, improve search engine rankings

Custom Web Development
Target: Companies with outdated, slow, or unattractive websites
Goal: Build modern websites using React or Python

AI Automation
Target: Companies with manual, repetitive processes
Goal: Build AI agents to automate tasks and save time
"""


scrape_tool = ScrapeWebsiteTool()


researcher = Agent(
    role="Market Researcher",
    goal="Analyze the target market website and identify their core market and potential weaknesses.",
    backstory="You are an expert at analyzing companies by reviewing their websites.",
    tools=[scrape_tool],
    verbose=True,
    memory=True,
    llm=gemini_llm
)

strategist = Agent(
    role="Agency Strategist",
    goal="Match the target market company needs with ONE of our agency services.",
    backstory=f"""You work for a top-tier digital agency. You must choose ONE best service from our offerings.

OUR SERVICES:
{agency_services}
""",
    verbose=True,
    memory=True,
    llm=gemini_llm
)

writer = Agent(
    role="Senior Sales Copywriter",
    goal="Write a personalized cold email that sounds human and professional.cold email should be easy to read even for kids",
    backstory="You write high-converting cold emails that feel personal and natural.",
    verbose=True,
    llm=gemini_llm
)


target_url = "https://openai.com/"

task_analyze = Task(
    description=f"""
Scrape the website {target_url}. Summarize what the company does and identify ONE key improvement area.
""",
    expected_output="Company summary and one clear pain point.",
    agent=researcher
)

task_strategize = Task(
    description="""Based on the research, select ONE agency service that best fits the company. Explain why.""",
    expected_output="Chosen service with reasoning.",
    agent=strategist
)

task_write = Task(
    description="""Write a cold email to the CEO. Pitch the chosen service. Keep it under 150 words.""",
    expected_output="A ready-to-send cold email that is easy to read even for kids.",
    agent=writer
)


sales_crew = Crew(
    agents=[researcher, strategist, writer],
    tasks=[task_analyze, task_strategize, task_write],
    verbose=True
)

print("STARTING SALES RESEARCH AGENT")
result = sales_crew.kickoff()

print("\nFINAL EMAIL DRAFT\n")
print(result)
