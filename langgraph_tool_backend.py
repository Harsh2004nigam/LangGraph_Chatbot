

## ✅ Full Updated Backend (ready to paste)


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
        "You are an AI assistant that can use tools.\n"
        "Available tools:\n"
        "1. search(query: str)\n"
        "2. calculator(first_num: float, second_num: float, operation: str)\n"
        "3. get_stock_price(symbol: str)\n\n"
        "If a user query requires a tool, respond ONLY with JSON in this format:\n"
        '{"tool": "tool_name", "args": {"arg1": "value", "arg2": "value"}}\n'
        "Do NOT add any extra text outside JSON. "
        "If no tool is needed, just answer normally."
    )

    response = model.invoke([HumanMessage(content=system_instruction)] + messages)
    text = response.content.strip()

    # Try to extract tool call
    tool_request = extract_json_from_text(text)

    # -------------------  TOOL EXECUTION UI STATUS  -------------------
    if tool_request and "tool" in tool_request:
        tool_name = tool_request.get("tool")
        args = tool_request.get("args", {})

        if tool_name in TOOLS:
            # Send an intermediate status message (for frontend)
            status_message = AIMessage(content=f"🔧 Using **{tool_name}** ...")
            
            # Actually execute the tool
            tool_fn = TOOLS[tool_name]
            result = tool_fn.invoke(args)
            
            # Simulate processing delay for UI realism
            time.sleep(1.5)
            
            # Then send final answer
            follow_up = (
                f"The tool '{tool_name}' returned this result: {result}. "
                f"Now give a short, clear final answer to the user."
            )
            final_response = model.invoke([HumanMessage(content=follow_up)])
            
            return {"messages": [status_message, AIMessage(content=final_response.content)]}

    # No tool used
    return {"messages": [response]}

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
