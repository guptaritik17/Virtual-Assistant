import os
import asyncio
import json
import re
from typing import Annotated, Sequence, List, TypedDict, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode 
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from preferences_data import categories, use_cases, important_features, known_brands


# ---------------------- Load environment ----------------------
load_dotenv()
groq_api_key = os.getenv("groq_api_key")

# ---------------------- LangGraph State ----------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_state: dict
    action: Literal["recommend", "clarify", "end"]
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

from rapidfuzz import process, fuzz


# ------------------------- HANDLE_QUERY FUNCTION -------------------------

async def handle_query(state: AgentState) -> AgentState:
    user_state = state.get("user_state", {})
    last_input = [msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)][-1]

    print("📩 User input:", last_input)
    print("🧠 Existing user_state before update:", user_state)

    query = last_input.lower()

    # ----------- Budget Extraction -----------
    budget_match = re.search(r'\b(?:under|within|upto|around)?\s*(\d{4,7})\s*(inr|rs|rupees)?', query)
    if budget_match:
        user_state["budget"] = budget_match.group(1)

    # ----------- Fuzzy Category Match -----------
    cat_match, score, _ = process.extractOne(query, categories, scorer=fuzz.partial_ratio)
    if score > 80:
        user_state["category"] = cat_match

    # ----------- Fuzzy Use Case Match -----------
    use_match, score, _ = process.extractOne(query, use_cases, scorer=fuzz.partial_ratio)

    if score > 80:
        user_state["use_case"] = use_match

    # ----------- Fuzzy Brand Preferences -----------
    for brand in known_brands:
        if re.search(rf'\b{re.escape(brand)}\b', query):
            if brand not in user_state["brand_preferences"]:
                user_state["brand_preferences"].append(brand)

    # ----------- Fuzzy Important Features -----------
    for feat in important_features:
        if fuzz.partial_ratio(feat, query) > 80 and feat not in user_state["important_features"]:
            user_state["important_features"].append(feat)

    # Ensure uniqueness
    user_state["brand_preferences"] = list(set(user_state["brand_preferences"]))
    user_state["important_features"] = list(set(user_state["important_features"]))

    print("🧠 Updated user_state after rules:", user_state)

    # ----------- Prompt Construction -----------
    formatted_prompt = prompt_template.format(
        user_input=last_input,
        category=user_state.get("category", "None"),
        budget=user_state.get("budget", "None"),
        use_case=user_state.get("use_case", "None"),
        brand_preferences=", ".join(user_state.get("brand_preferences", [])) or "None",
        important_features=", ".join(user_state.get("important_features", [])) or "None"
    )
    print("\n📝 Final prompt to model:\n", formatted_prompt)

    try:
        response = await model.ainvoke([
            SystemMessage(content=formatted_prompt),
            *state["messages"]
        ])
        assistant_reply = response.content.strip()
    except Exception as e:
        print("❌ Error invoking model:", e)
        assistant_reply = "I'm having trouble responding right now. Please try again."
        response = SystemMessage(content=assistant_reply)

    if not assistant_reply:
        assistant_reply = "Sorry, I didn't understand that. Can you rephrase?"

    print("🤖 Assistant reply:", assistant_reply)

    updated_messages = state["messages"] + [response]

    # ----------- Decide Action -----------
    action = "recommend" if "here are some" in assistant_reply.lower() else "clarify"
    print("🚦 Next action:", action)

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
