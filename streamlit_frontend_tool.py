import streamlit as st
from langgraph_tool_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
import json

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id, "New Chat")
    st.session_state["message_history"] = []

def add_thread(thread_id, title):
    if "chat_threads" not in st.session_state:
        st.session_state["chat_threads"] = []
    if "chat_titles" not in st.session_state:
        st.session_state["chat_titles"] = {}

    thread_str = str(thread_id)
    if thread_str not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_str)
    st.session_state["chat_titles"].setdefault(thread_str, title or "New Chat")

def load_conversation(thread_id):
    thread_str = str(thread_id)
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_str}})
    return state.values.get("messages", [])

def delete_thread(thread_id):
    thread_str = str(thread_id)
    if "chat_threads" in st.session_state and thread_str in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].remove(thread_str)
    if "chat_titles" in st.session_state:
        st.session_state["chat_titles"].pop(thread_str, None)
    if st.session_state.get("thread_id") == thread_str:
        reset_chat()

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

backend_threads = retrieve_all_threads() or []
backend_threads = [str(t) for t in backend_threads]

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = backend_threads.copy()

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

for t in st.session_state["chat_threads"]:
    st.session_state["chat_titles"].setdefault(t, "New Chat")

add_thread(st.session_state["thread_id"], "New Chat")

# ============================ Sidebar ============================
st.sidebar.title("All Chats")

if st.sidebar.button("🆕 New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("💬 My Conversations")

st.markdown("""
    <style>
    .sidebar-chat {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #2d2d2d;
        padding: 8px 12px;
        border-radius: 8px;
        color: white;
        margin-bottom: 8px;
        transition: 0.12s;
    }
    .sidebar-chat:hover { background-color: #3d3d3d; }
    .delete-btn {
        color: #ff6b6b;
        border: none;
        background: transparent;
        font-size: 18px;
        cursor: pointer;
    }
    .delete-btn:hover { color: #ff4d4d; }
    </style>
""", unsafe_allow_html=True)

for thread_id in st.session_state["chat_threads"][::-1]:
    thread_str = str(thread_id)
    title = st.session_state["chat_titles"].get(thread_str, "New Chat")

    if title in ("New Chat", "", None):
        try:
            msgs = load_conversation(thread_str)
            first_user = None
            for m in msgs:
                typ = getattr(m, "type", "").lower() if hasattr(m, "type") else m.get("type", "").lower() if isinstance(m, dict) else ""
                if typ == "human" or getattr(m, "role", "") == "user":
                    first_user = getattr(m, "content", m.get("content") if isinstance(m, dict) else "")
                    break
            if first_user:
                new_title = first_user.strip()[:45] + ("..." if len(first_user.strip()) > 45 else "")
                st.session_state["chat_titles"][thread_str] = new_title
                title = new_title
        except Exception:
            pass

    short_title = title if len(title) <= 45 else title[:45] + "..."

    col1, col2 = st.sidebar.columns([0.85, 0.15])
    with col1:
        if st.button(short_title, key=f"select_{thread_str}"):
            st.session_state["thread_id"] = thread_str
            msgs = load_conversation(thread_str)
            st.session_state["message_history"] = [
                {"role": "user" if getattr(m, "type", "").lower() == "human" or getattr(m, "role", "") == "user" else "assistant",
                 "content": getattr(m, "content", m.get("content") if isinstance(m, dict) else "")}
                for m in msgs
            ]
            st.rerun()
    with col2:
        if st.button("🗑", key=f"delete_{thread_str}"):
            delete_thread(thread_str)
            st.rerun()

# ============================ Main Chat ============================
st.title("🤖 LangGraph Chatbot")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    thread_id = st.session_state["thread_id"]
    current_title = st.session_state["chat_titles"].get(thread_id, "New Chat")

    if current_title in ("New Chat", "", None):
        trimmed = user_input.strip()
        new_title = trimmed[:45] + ("..." if len(trimmed) > 45 else "")
        st.session_state["chat_titles"][thread_id] = new_title

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        status_box = None

        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):
            # Show tool activity only in a clean way
            if isinstance(message_chunk, ToolMessage):
                tool_name = getattr(message_chunk, "name", "tool")
                if status_box is None:
                    status_box = st.status(f"🔧 Using {tool_name} ...", expanded=False)
                else:
                    status_box.update(label=f"🔧 Using {tool_name} ...", state="running")
                continue

            if isinstance(message_chunk, AIMessage):
                content = message_chunk.content.strip()
                # Skip JSON-like tool messages
                if content.startswith("{") and content.endswith("}"):
                    continue
                try:
                    # Try parsing JSON and ignore if valid
                    json.loads(content)
                    continue
                except:
                    pass

                # Clean normal output
                full_response += content + " "
                response_placeholder.markdown(full_response.strip())

    st.session_state["message_history"].append({"role": "assistant", "content": full_response.strip()})








































# import streamlit as st
# from langgraph_tool_backend import chatbot, retrieve_all_threads
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# import uuid

# # =========================== Utilities ===========================
# def generate_thread_id():
#     return uuid.uuid4()

# def reset_chat():
#     thread_id = generate_thread_id()
#     st.session_state["thread_id"] = thread_id
#     add_thread(thread_id)
#     st.session_state["message_history"] = []

# def add_thread(thread_id):
#     if thread_id not in st.session_state["chat_threads"]:
#         st.session_state["chat_threads"].append(thread_id)

# def load_conversation(thread_id):
#     state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
#     return state.values.get("messages", [])

# # ======================= Session Initialization ===================
# if "message_history" not in st.session_state:
#     st.session_state["message_history"] = []

# if "thread_id" not in st.session_state:
#     st.session_state["thread_id"] = generate_thread_id()

# if "chat_threads" not in st.session_state:
#     st.session_state["chat_threads"] = retrieve_all_threads()

# add_thread(st.session_state["thread_id"])

# # ============================ Sidebar ============================
# st.sidebar.title("LangGraph Chatbot (Hugging Face + Tools)")

# if st.sidebar.button("🆕 New Chat"):
#     reset_chat()

# st.sidebar.header("💬 My Conversations")
# for thread_id in st.session_state["chat_threads"][::-1]:
#     if st.sidebar.button(str(thread_id)):
#         st.session_state["thread_id"] = thread_id
#         messages = load_conversation(thread_id)

#         temp_messages = []
#         for msg in messages:
#             role = "user" if isinstance(msg, HumanMessage) else "assistant"
#             temp_messages.append({"role": role, "content": msg.content})
#         st.session_state["message_history"] = temp_messages

# # ============================ Main UI ============================

# st.title("🤖 AI Assistant with Tools (Hugging Face Backend)")

# # Render previous messages
# for message in st.session_state["message_history"]:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Chat input
# user_input = st.chat_input("Type your message...")

# if user_input:
#     # Show user message
#     st.session_state["message_history"].append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.markdown(user_input)

#     CONFIG = {
#         "configurable": {"thread_id": st.session_state["thread_id"]},
#         "metadata": {"thread_id": st.session_state["thread_id"]},
#         "run_name": "chat_turn",
#     }

#     # Assistant message container
#     with st.chat_message("assistant"):
#         response_placeholder = st.empty()
#         full_response = ""
#         status_box = None

#         # Stream responses
#         for message_chunk, metadata in chatbot.stream(
#             {"messages": [HumanMessage(content=user_input)]},
#             config=CONFIG,
#             stream_mode="messages",
#         ):
#             # Show clean status container for tools
#             if isinstance(message_chunk, ToolMessage):
#                 tool_name = getattr(message_chunk, "name", "tool")
#                 if status_box is None:
#                     status_box = st.status(f"🔧 Using `{tool_name}`...", expanded=True)
#                 else:
#                     status_box.update(
#                         label=f"🔧 Using `{tool_name}`...",
#                         state="running",
#                         expanded=True,
#                     )

#             # Stream assistant message tokens
#             elif isinstance(message_chunk, AIMessage):
#                 full_response += message_chunk.content
#                 response_placeholder.markdown(full_response)

#         # Finalize status box if tool was used
#         if status_box is not None:
#             status_box.update(label="✅ Tool finished", state="complete", expanded=False)

#         # Show final clean answer below
#         response_placeholder.markdown(full_response.strip())

#     # Save assistant message
#     st.session_state["message_history"].append(
#         {"role": "assistant", "content": full_response.strip()}
#     )
































# import streamlit as st
# from langgraph_tool_backend import chatbot, retrieve_all_threads
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# import uuid

# # =========================== Utilities ===========================
# def generate_thread_id():
#     return uuid.uuid4()

# def reset_chat():
#     thread_id = generate_thread_id()
#     st.session_state["thread_id"] = thread_id
#     add_thread(thread_id)
#     st.session_state["message_history"] = []

# def add_thread(thread_id):
#     if thread_id not in st.session_state["chat_threads"]:
#         st.session_state["chat_threads"].append(thread_id)

# def load_conversation(thread_id):
#     state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
#     # Check if messages key exists in state values, return empty list if not
#     return state.values.get("messages", [])

# # ======================= Session Initialization ===================
# if "message_history" not in st.session_state:
#     st.session_state["message_history"] = []

# if "thread_id" not in st.session_state:
#     st.session_state["thread_id"] = generate_thread_id()

# if "chat_threads" not in st.session_state:
#     st.session_state["chat_threads"] = retrieve_all_threads()

# add_thread(st.session_state["thread_id"])

# # ============================ Sidebar ============================
# st.sidebar.title("LangGraph Chatbot")

# if st.sidebar.button("New Chat"):
#     reset_chat()

# st.sidebar.header("My Conversations")
# for thread_id in st.session_state["chat_threads"][::-1]:
#     if st.sidebar.button(str(thread_id)):
#         st.session_state["thread_id"] = thread_id
#         messages = load_conversation(thread_id)

#         temp_messages = []
#         for msg in messages:
#             role = "user" if isinstance(msg, HumanMessage) else "assistant"
#             temp_messages.append({"role": role, "content": msg.content})
#         st.session_state["message_history"] = temp_messages

# # ============================ Main UI ============================

# # Render history
# for message in st.session_state["message_history"]:
#     with st.chat_message(message["role"]):
#         st.text(message["content"])

# user_input = st.chat_input("Type here")

# if user_input:
#     # Show user's message
#     st.session_state["message_history"].append({"role": "user", "content": user_input})
#     with st.chat_message("user"):
#         st.text(user_input)

#     CONFIG = {
#         "configurable": {"thread_id": st.session_state["thread_id"]},
#         "metadata": {"thread_id": st.session_state["thread_id"]},
#         "run_name": "chat_turn",
#     }

#     # Assistant streaming block
#     with st.chat_message("assistant"):
#         # Use a mutable holder so the generator can set/modify it
#         status_holder = {"box": None}

#         def ai_only_stream():
#             for message_chunk, metadata in chatbot.stream(
#                 {"messages": [HumanMessage(content=user_input)]},
#                 config=CONFIG,
#                 stream_mode="messages",
#             ):
#                 # Lazily create & update the SAME status container when any tool runs
#                 if isinstance(message_chunk, ToolMessage):
#                     tool_name = getattr(message_chunk, "name", "tool")
#                     if status_holder["box"] is None:
#                         status_holder["box"] = st.status(
#                             f"🔧 Using `{tool_name}` …", expanded=True
#                         )
#                     else:
#                         status_holder["box"].update(
#                             label=f"🔧 Using `{tool_name}` …",
#                             state="running",
#                             expanded=True,
#                         )

#                 # Stream ONLY assistant tokens
#                 if isinstance(message_chunk, AIMessage):
#                     yield message_chunk.content

#         ai_message = st.write_stream(ai_only_stream())

#         # Finalize only if a tool was actually used
#         if status_holder["box"] is not None:
#             status_holder["box"].update(
#                 label="✅ Tool finished", state="complete", expanded=False
#             )

#     # Save assistant message
#     st.session_state["message_history"].append(
#         {"role": "assistant", "content": ai_message}
#     )
