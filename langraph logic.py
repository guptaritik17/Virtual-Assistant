import os
import asyncio
import json
import re
from typing import Annotated, Sequence, List, TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from cerberus import Validator

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode 
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# ---------------------- Load environment ----------------------
load_dotenv()
groq_api_key = os.getenv("groq_api_key")

# ---------------------- LangGraph State ----------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_state: dict
    action: Literal["recommend", "clarify", "end"]

# ---------------------- Pydantic Preferences Model ----------------------
class Preferences(BaseModel):
    budget: str | None = None
    use_case: str | None = None
    category: str | None = None
    brand_preferences: list[str] = []
    important_features: list[str] = []
    excluded_features: list[str] = []

# ---------------------- Cerberus Schema ----------------------
schema = {
    "budget": {"type": "string", "nullable": True},
    "use_case": {"type": "string", "nullable": True},
    "category": {"type": "string", "nullable": True},
    "brand_preferences": {"type": "list", "schema": {"type": "string"}},
    "important_features": {"type": "list", "schema": {"type": "string"}},
    "excluded_features": {"type": "list", "schema": {"type": "string"}},
}
validator = Validator(schema)

# ---------------------- Dummy Tool (Optional) ----------------------
@tool
def search_products(category: str, budget: str, use_case: str, brand_preferences: List[str]) -> str:
    """
    Simulates fetching products from a dummy database based on user preferences.
    """
    return f"Fetched products for category: {category}, budget: {budget}, use_case: {use_case}, brands: {brand_preferences}"

tools = [search_products]

# ---------------------- LLM Setup ----------------------
model = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192").bind_tools(tools)

# ---------------------- Prompt Template ----------------------
prompt_template = """
You are a smart shopping assistant for Walmart.

Before updating or acting on any preferences:
- First assess whether the user input is clear, realistic, and consistent with the preferences collected so far.
- Do NOT update state blindly. Only update if the input is valid and makes sense for the use case.
- If the budget or expectations are too low or contradictory (e.g., gaming laptop under ₹20,000), politely clarify or suggest reasonable options.
- Avoid asking again for any information already collected.

User said: "{user_input}"

Current known preferences:
- Category: {category}
- Budget: {budget}
- Use case: {use_case}
- Preferred brands: {brand_preferences}
- Important features: {important_features}

Your tasks:
1. If needed, ask a clarifying question.
2. If the preferences are realistic and sufficient (i.e., category, budget, and use case are valid), recommend suitable products.
3. Always explain any technical terms you use.
4. Do not repeat questions for known preferences.
5. At the end, return an `action` field with either "recommend" or "clarify" to guide the assistant’s next step.
"""


# ---------------------- Node 1: handle_query ----------------------

async def handle_query(state: AgentState) -> AgentState:
    user_state = state.get("user_state", {})
    last_input = [msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]

    formatted_prompt = prompt_template.format(
        user_input=last_input,
        category=user_state.get("category", "None"),
        budget=user_state.get("budget", "None"),
        use_case=user_state.get("use_case", "None"),
        brand_preferences=", ".join(user_state.get("brand_preferences", [])) or "None",
        important_features=", ".join(user_state.get("important_features", [])) or "None"
    )
    print("Formatted prompt:", formatted_prompt)  # Debug print

    response = await model.ainvoke([
        SystemMessage(content=formatted_prompt),
        *state["messages"]
    ])
    assistant_reply = response.content
    print("Assistant raw reply:", assistant_reply)  # Debug print

    # Update chat history
    updated_messages = state["messages"] + [response]

    # ---------------------- Extract Updated State ----------------------
    state_extraction_prompt = f"""
You are a state extraction system.
User: "{last_input}"
Assistant: "{assistant_reply}"

Extract preferences as JSON:
{{
  "budget": "...",
  "use_case": "...",
  "category": "...",
  "brand_preferences": ["..."],
  "important_features": ["..."],
  "excluded_features": ["..."]
}}
If nothing new, return {{}}
"""
    extraction = await model.ainvoke([SystemMessage(content=state_extraction_prompt)])
    try:
        raw = extraction.content
        print("Raw extraction output:", raw)  # Debug print
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        updates = {}
        if match:
            updates = json.loads(match.group())
            # Validate with Cerberus
            if validator.validate(updates):
                user_state.update(validator.document)
            else:
                print("Validation error:", validator.errors)
            # Validate with Pydantic (optional, for stricter schema)
            try:
                prefs = Preferences(**{**user_state})
                user_state.update(prefs.dict(exclude_unset=True))
            except ValidationError as e:
                print("Pydantic validation error:", e)
            # Fallback: Only update fields that are present and not empty/None/"..."
            for key, value in updates.items():
                if value is not None and value != "..." and not (isinstance(value, str) and value.strip() == "..."):
                    user_state[key] = value

        # Fallback: Try to extract common fields from user input if LLM fails
        if "budget" in last_input.lower() and ("budget" not in updates or updates.get("budget") in [None, "...", ""]):
            budget_match = re.search(r'(\d{4,7})\s*(inr|rs|rupees)?', last_input.lower())
            if budget_match:
                user_state["budget"] = budget_match.group(1)
        # Add similar fallback extraction for other fields as needed

    except Exception as e:
        print("State extraction error:", e)

    # ---------------------- Decide Next Step ----------------------
    action = "recommend" if "Here are some" in assistant_reply else "clarify"

    return {
        **state,
        "messages": updated_messages,
        "user_state": user_state,
        "action": action
    }

# ---------------------- Conditional Branch ----------------------
def should_continue(state: AgentState) -> Literal["tool", "generate"]:
    return "tool" if state["action"] == "recommend" else "generate"

# ---------------------- Node 2: tool execution ----------------------
tool_node = ToolNode(tools=tools)

# ---------------------- Node 3: final response generation ----------------------
async def generate_response(state: AgentState) -> AgentState:
    prompt = "Respond helpfully based on current user preferences and your previous conversation."
    response = await model.ainvoke([
        SystemMessage(content=prompt),
        *state["messages"]
    ])
    return {
        **state,
        "messages": state["messages"] + [response]
    }

# ---------------------- LangGraph Flow ----------------------
graph = StateGraph(AgentState)
graph.set_entry_point("handle_query")
graph.add_node("handle_query", handle_query)
graph.add_node("tool_execution", tool_node)
graph.add_node("generate_response", generate_response)
graph.add_conditional_edges("handle_query", should_continue, {
    "tool": "tool_execution",
    "generate": "generate_response"
})
graph.add_edge("tool_execution", "generate_response")
graph.add_edge("generate_response", END)
app = graph.compile()

# ---------------------- Run the Agent ----------------------
async def run_assistant():
    print("🛒 Smart Walmart Assistant (Type 'stop' to quit)\n")
    user_state = {
        "category": None,
        "budget": None,
        "use_case": None,
        "brand_preferences": [],
        "important_features": [],
        "excluded_features": [],
        "suggested_products": [],
        "chat_history": []
    }

    while True:
        query = input("👤 You: ")
        if query.lower() in ["stop", "exit", "quit"]:
            break
        inputs = {
            "messages": [HumanMessage(content=query)],
            "user_state": user_state,
            "action": "clarify"
        }
        result = await app.ainvoke(inputs)
        final_msg = result["messages"][-1].content
        print(f"\n🤖 Assistant:\n{final_msg}\n")

# ---------------------- Entry ----------------------
if __name__ == "__main__":
    asyncio.run(run_assistant())
