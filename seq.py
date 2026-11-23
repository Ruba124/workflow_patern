# Cell [1]
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field


load_dotenv()

# Initialize LLM with OpenRouter free model
llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)



class InfoCheck(BaseModel):
    is_complete: bool = Field(
        description="True if we have enough concrete technical details to write a strong CV without placeholders; False if critical details (like tech stack or degree/projects) are missing."
    )
    follow_up_question: str | None = Field(
        default=None,
        description="A single, specific question to ask the user if is_complete is False. Otherwise None."
    )


class FormattedCV(BaseModel):
    headline: str = Field(description="A crisp 1-line LinkedIn/CV headline (no placeholders)")
    professional_summary: str = Field(description="A clean 3-sentence summary paragraph with no brackets or placeholders")
    core_skills: list[str] = Field(description="List of 4-6 relevant technical skills")
    career_objective: str = Field(description="One direct sentence stating the target role")

class State(TypedDict):
    name: str
    job: str
    skill: str
    goal: str
    raw_cv: str
    structured_cv: Optional[FormattedCV]  # <-- New field for the Pydantic object
    is_complete: bool
    follow_up_question: str | None
    question_count: int

# Cell [7]
def ask_name(state: State):
    name = input("What's your full name? ")
    return {"name": name}

# Cell [8]
def ask_job(state: State):
    job = input("💼 What is your most recent job title (or 'student' if no job)? ")
    return {"job": job}

# Cell [9]
def ask_skill(state: State):
    skill = input("🛠️ What is your top professional skill? ")
    return {"skill": skill}

# Cell [10]
def ask_goal(state: State):
    goal = input("🎯 What's your next career goal or desired job? ")
    return {"goal": goal}

# Cell [11]
def assemble_cv(state: State):
    raw = (
        f"Name: {state['name']}\n"
        f"Recent Position: {state['job']}\n"
        f"Top Skill: {state['skill']}\n"
        f"Career Goal: {state['goal']}\n"
    )
    print("\n📄 [PROMPT TO POLISH CV NODE WILL BE THIS RAW CV TEXT]\n", raw)
    return {**state, "raw_cv": raw}

# Cell [14]
def output_cv(state: State):
    cv = state["structured_cv"]
    print("\n" + "=" * 45)
    print(f"👤 {state['name'].title()} | {cv.headline}")
    print("=" * 45)
    print(f"\n📌 SUMMARY:\n{cv.professional_summary}")
    print(f"\n🛠️ CORE SKILLS:\n• " + " • ".join(cv.core_skills))
    print(f"\n🎯 OBJECTIVE:\n{cv.career_objective}")
    print("=" * 45 + "\n")
    return state

# Cell [12]
def polish_cv(state: State):
    structured_llm = llm.with_structured_output(FormattedCV)
    
    prompt = (
        "Take this basic CV info and polish it into clean, professional structured fields. "
        "Do NOT include markdown symbols or bracketed placeholders like [University]:\n\n"
        f"{state['raw_cv']}"
    )
    
    cv_object = structured_llm.invoke(prompt)
    return {"structured_cv": cv_object}
def check_missing_info(state: State):
    count = state.get("question_count", 0)
    
    # Hard safety brake: never ask more than 2 follow-up questions
    if count >= 2:
        return {"is_complete": True, "follow_up_question": None}

    checker_llm = llm.with_structured_output(InfoCheck)
    prompt = (
        "Review this raw CV info. If it lacks specific programming languages, tools, or academic/project context "
        "needed to avoid bracketed placeholders, set is_complete=False and ask ONE concise question.\n\n"
        f"{state['raw_cv']}"
    )
    result = checker_llm.invoke(prompt)
    return {
        "is_complete": result.is_complete,
        "follow_up_question": result.follow_up_question,
    }

def ask_followup(state: State):
    answer = input(f"\n🤔 [AI Follow-up]: {state['follow_up_question']} ")
    updated_raw = state["raw_cv"] + f"Additional Details: {answer}\n"
    return {
        "raw_cv": updated_raw,
        "question_count": state.get("question_count", 0) + 1,
    }

def route_after_check(state: State) -> str:
    if state["is_complete"]:
        return "polish_cv"
    return "ask_followup"
# Cell [15]
graph = StateGraph(State)
graph.add_node("ask_name", ask_name)
graph.add_node("ask_job", ask_job)
graph.add_node("ask_skill", ask_skill)
graph.add_node("ask_goal", ask_goal)
graph.add_node("assemble_cv", assemble_cv)
graph.add_node("polish_cv", polish_cv)
graph.add_node("output", output_cv)
graph.add_node("check_missing_info", check_missing_info)
graph.add_node("ask_followup", ask_followup)



# Cell [16]
graph.add_edge(START, "ask_name")
graph.add_edge("ask_name", "ask_job")
graph.add_edge("ask_job", "ask_skill")
graph.add_edge("ask_skill", "ask_goal")
graph.add_edge("ask_goal", "assemble_cv")
graph.add_edge("assemble_cv", "check_missing_info")

graph.add_conditional_edges(
    "check_missing_info",
    route_after_check,
    {
        "ask_followup": "ask_followup",
        "polish_cv": "polish_cv",
    },
)

# Loop back from ask_followup to check_missing_info!
graph.add_edge("ask_followup", "check_missing_info")

graph.add_edge("polish_cv", "output")
graph.add_edge("output", END)

app = graph.compile()
png_bytes = app.get_graph().draw_mermaid_png()
with open("graph_diagram.png", "wb") as f:
    f.write(png_bytes)

# 2. Actually run the graph
final_state = app.invoke({})

