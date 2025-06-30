import os
import asyncio
from dotenv import load_dotenv
from typing import Annotated, Sequence, TypedDict, Literal, List
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.tools import tool

load_dotenv()
groq_api_key = os.getenv("groq_api_key")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context_docs: List[str]  # from context VectorDB
    order_docs: List[str]    # from order history VectorDB

async def main():
    client = MultiServerMCPClient(
        {
            "product_catalog_fetch": {
                "command": "python",
                "args": ["C:/Users/priti/OneDrive/Desktop/Walmart/Virtual-Assistant/server/ProductCatalogFetchServer.py"],
                "transport": "stdio",
            },
        }
    )
    tools = await client.get_tools()
    print("Fetched tools:", tools)

    model = ChatGroq(
        groq_api_key=groq_api_key,
        model_name='gemma2-9b-it'
    ).bind_tools(tools)

    async def handle_query(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(content="You are a shopping assistant that collects data from an api call reponse using tools. Your job is to fetch the data from the API by choosing the right one from the user query and return the results as a string.")
        response = await model.ainvoke([system_prompt] + state["messages"])
        print("\nAgent Response:", response, "\ntype of response is: ", type(response))
        return {**state, "messages": state["messages"] + [response]}

    def should_continue(state: AgentState) -> Literal["end", "continue"]:
        last_msg = state["messages"][-1]
        return "continue" if getattr(last_msg, "tool_calls", None) else "end"

    async def generate_response(state: AgentState) -> AgentState:
        """
        Combines catalog data (via tool), context docs, and order history docs to generate a final, helpful answer.
        """
        full_context = "\n\n".join(state.get("context_docs", []))
        order_context = "\n\n".join(state.get("order_docs", []))

        context_prompt = f"""
        You are a helpful shopping assistant. Based on the user query, the product catalog information retrieved from an API, the user's chat context, and their order history, suggest the most appropriate products and explain your reasoning.

        Context:
        {full_context}

        Order History:
        {order_context}

        Conversation:
        """

        model_response = await model.ainvoke([
            SystemMessage(content=context_prompt),
            *state["messages"]
        ])

        return {**state, "messages": state["messages"] + [model_response]}

    graph = StateGraph(AgentState)
    graph.set_entry_point("handle_query")
    graph.add_node("handle_query", handle_query)
    graph.add_node("tool_execution", ToolNode(tools=tools))
    graph.add_node("generate_response", generate_response)

    graph.add_conditional_edges(
        "handle_query",
        should_continue,
        {
            "continue": "tool_execution",
            "end": "generate_response"
        }
    )

    graph.add_edge("tool_execution", "generate_response")
    graph.add_edge("generate_response", END)

    app = graph.compile()

    inputQuery = "What is the best electronics product for a laptop?"

    inputs = {
        "messages": [HumanMessage(content=inputQuery)],
        "context_docs": [],  
        "order_docs": []     
    }

    result = await app.ainvoke(inputs)
    print("\nFinal Response:", result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())