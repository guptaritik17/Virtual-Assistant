import os
from dotenv import load_dotenv
import google.generativeai as genai
import re
import json

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize user state
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

print("Welcome to the Walmart Shopping Assistant! (Type 'stop' to end the conversation.)")

while True:
    user_input = input("👤 You: ")
    if user_input.strip().lower() in ["stop", "exit", "quit"]:
        print("👋 Conversation ended.")
        break

    # Build prompt with current state
    prompt = f"""
IMPORTANT: You MUST always explain every technical term or jargon you use, immediately after mentioning it, in simple language.
For example: "This is a top load washing machine (which means you load clothes from the top)."

You are a smart shopping assistant for Walmart.
The user said: "{user_input}"

Their current preferences:
- Category: {user_state['category']}
- Budget: {user_state['budget']}
- Use case: {user_state['use_case']}
- Preferred brands: {', '.join(user_state['brand_preferences'])}
- Important features: {', '.join(user_state['important_features'])}

Based on this, do one of the following:
1. Ask a clarifying question if needed.
2. Recommend products (example-based).
3. Explain any technical terms you use.
4. Update their preferences if something new was mentioned.

Respond like a helpful assistant.
"""

    try:
        response = model.generate_content(prompt)
        assistant_reply = response.text
    except Exception as e:
        print(f"❌ API Error: {e}")
        break

    print(f"\n🛍️ Assistant:\n{assistant_reply}")

    # Update chat history
    user_state["chat_history"].append({
        "user": user_input,
        "assistant": assistant_reply
    })

    # Extract new preferences from the conversation
    state_extraction_prompt = f"""
You are an intelligent state extraction system.
Based on the user input and assistant reply, extract any new preferences.

User: "{user_input}"
Assistant: "{assistant_reply}"

Return only the updated fields in this JSON format:
{{
  "budget": "...",
  "use_case": "...",
  "category": "...",
  "brand_preferences": ["..."],
  "important_features": ["..."],
  "excluded_features": ["..."]

}}
If nothing was mentioned, return an empty JSON object: {{}}
"""
    state_response = model.generate_content(state_extraction_prompt)
    raw = state_response.text
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        extracted = json.loads(match.group())
        for key, value in extracted.items():
            if key in user_state and value not in [None, "", [], {}]:
                user_state[key] = value
    else:
        print("No JSON found in response.")

    # Optionally, print updated state for debugging
    # print("\n📦 Updated User State:")
    # print(json.dumps(user_state, indent=2))


# Finally, print the user state
print("\n📦 Final User State:")
print(json.dumps(user_state, indent=2))