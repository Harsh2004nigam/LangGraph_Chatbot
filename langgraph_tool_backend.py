

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import sqlite3
import requests
import json
import re
import time

load_dotenv()

# ------------------- 1. LLM -------------------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.3,
    max_new_tokens=512,
)
model = ChatHuggingFace(llm=llm)

# ------------------- 2. Tools -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform basic arithmetic operations."""
    try:
        if operation == "add":
            return {"result": first_num + second_num}
        elif operation == "sub":
            return {"result": first_num - second_num}
        elif operation == "mul":
            return {"result": first_num * second_num}
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero"}
            return {"result": first_num / second_num}
        else:
            return {"error": "Invalid operation"}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price from AlphaVantage."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=FGW5D4GS08IPJIVA"
    r = requests.get(url)
    data = r.json()
    try:
        return {"price": float(data["Global Quote"]["05. price"])}
    except Exception:
        return {"error": "Could not fetch stock price"}

TOOLS = {
    "search": search_tool,
    "calculator": calculator,
    "get_stock_price": get_stock_price,
}

# ------------------- 3. State -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ------------------- 4. Utility -------------------
def extract_json_from_text(text: str):
    """Extract the first valid JSON block from mixed model output."""
    json_pattern = re.compile(r"\{.*\}", re.DOTALL)
    match = json_pattern.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        cleaned = match.group().replace("'", '"')
        try:
            return json.loads(cleaned)
        except:
            return None

# ------------------- 5. Chat Node -------------------
def chat_node(state: ChatState):
    messages = state["messages"]

    system_instruction = (
        "You are an intelligent AI assistant that can use tools.\n"
        "Available tools:\n"
        "- search(query: str)\n"
        "- calculator(first_num: float, second_num: float, operation: str)\n"
        "- get_stock_price(symbol: str)\n\n"
        "If a query *requires* a tool, respond ONLY with valid JSON like:\n"
        '{"tool": "tool_name", "args": {"arg1": "value", "arg2": "value"}}\n'
        "Otherwise, answer normally in plain text.\n"
        "Never include explanations, reasoning, or markdown inside tool JSON.\n"
        "Keep your answers conversational, clear, and human-friendly."
    )

    response = model.invoke([HumanMessage(content=system_instruction)] + messages)
    text = response.content.strip()

    # Try to extract tool call
    tool_request = extract_json_from_text(text)

    # ------------------- TOOL EXECUTION -------------------
    if tool_request and "tool" in tool_request:
        tool_name = tool_request.get("tool")
        args = tool_request.get("args", {})

        if tool_name in TOOLS:
            # Step 1: Send "Using tool..." status to frontend
            status_message = AIMessage(content=f"🔧 Using **{tool_name}**...")

            # Step 2: Execute tool
            tool_fn = TOOLS[tool_name]
            result = tool_fn.invoke(args)

            # Step 3: Handle result and possible errors
            if not result or "error" in result:
                error_msg = result.get("error", "Tool failed to return a valid response.")
                return {
                    "messages": [
                        status_message,
                        AIMessage(content=f"⚠️ The {tool_name} tool encountered an error: {error_msg}. Please recheck your input.")
                    ]
                }

            # Step 4: Ask LLM to summarize the tool result nicely
            time.sleep(1)
            follow_up_prompt = (
                f"The tool '{tool_name}' returned this result:\n{result}\n\n"
                "Now, write a short, clear, and natural final answer for the user. "
                "Do not include JSON or tool names, just explain the result nicely."
            )
            final_response = model.invoke([HumanMessage(content=follow_up_prompt)])
            final_answer = final_response.content.strip()

            return {"messages": [status_message, AIMessage(content=final_answer)]}

    # ------------------- NO TOOL -------------------
    if text.startswith("{") and text.endswith("}"):
        return {"messages": [AIMessage(content="Sorry, I couldn’t process that request.")]}
    
    return {"messages": [AIMessage(content=text)]}

# ------------------- 6. Checkpointer -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ------------------- 7. Graph -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

# ------------------- 8. Helper -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)































# # backend.py

# from langgraph.graph import StateGraph, START, END
# from typing import TypedDict, Annotated
# from langchain_core.messages import BaseMessage, HumanMessage
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langgraph.checkpoint.sqlite import SqliteSaver
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_community.tools import DuckDuckGoSearchRun
# from langchain_core.tools import tool
# from dotenv import load_dotenv
# import sqlite3
# import requests

# load_dotenv()

# # -------------------
# # 1. LLM
# # -------------------
# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     # repo_id="google/gemma-2-2b-it",
#     task="text-generation"
# )
# model = ChatHuggingFace(llm=llm)
# # -------------------
# # 2. Tools
# # -------------------
# # Tools
# search_tool = DuckDuckGoSearchRun(region="us-en")

# @tool
# def calculator(first_num: float, second_num: float, operation: str) -> dict:
#     """
#     Perform a basic arithmetic operation on two numbers.
#     Supported operations: add, sub, mul, div
#     """
#     try:
#         if operation == "add":
#             result = first_num + second_num
#         elif operation == "sub":
#             result = first_num - second_num
#         elif operation == "mul":
#             result = first_num * second_num
#         elif operation == "div":
#             if second_num == 0:
#                 return {"error": "Division by zero is not allowed"}
#             result = first_num / second_num
#         else:
#             return {"error": f"Unsupported operation '{operation}'"}
        
#         return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
#     except Exception as e:
#         return {"error": str(e)}




# @tool
# def get_stock_price(symbol: str) -> dict:
#     """
#     Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
#     using Alpha Vantage with API key in the URL.
#     """
#     url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
#     r = requests.get(url)
#     return r.json()



# tools = [search_tool, get_stock_price, calculator]
# llm_with_tools = model.bind_tools(tools)

# # -------------------
# # 3. State
# # -------------------
# class ChatState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

# # -------------------
# # 4. Nodes
# # -------------------
# def chat_node(state: ChatState):
#     """LLM node that may answer or request a tool call."""
#     messages = state["messages"]
#     response = llm_with_tools.invoke(messages)
#     return {"messages": [response]}

# tool_node = ToolNode(tools)

# # -------------------
# # 5. Checkpointer
# # -------------------
# conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
# checkpointer = SqliteSaver(conn=conn)

# # -------------------
# # 6. Graph
# # -------------------
# graph = StateGraph(ChatState)
# graph.add_node("chat_node", chat_node)
# graph.add_node("tools", tool_node)

# graph.add_edge(START, "chat_node")

# graph.add_conditional_edges("chat_node",tools_condition)
# graph.add_edge('tools', 'chat_node')

# chatbot = graph.compile(checkpointer=checkpointer)

# # -------------------
# # 7. Helper
# # -------------------
# def retrieve_all_threads():
#     all_threads = set()
#     for checkpoint in checkpointer.list(None):
#         all_threads.add(checkpoint.config["configurable"]["thread_id"])
#     return list(all_threads)
