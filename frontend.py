import streamlit as st
from project_4 import agent,retrive_all_threads
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import uuid

def gen_thread_id():
    return uuid.uuid4()

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_thread_ids"]:
        st.session_state["chat_thread_ids"].append(thread_id)

def reset_chat():
    id = gen_thread_id()
    st.session_state['thread_id'] = id
    add_thread(st.session_state['thread_id'])
    st.session_state['history'] = []

def load_conv(thread_id):
    state = agent.get_state(config={'configurable':{"thread_id":thread_id}})
    return state.values.get('messages',[])

if "history" not in st.session_state:
    st.session_state['history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = gen_thread_id()

if 'chat_thread_ids' not in st.session_state:
    st.session_state['chat_thread_ids'] = retrive_all_threads()

add_thread(st.session_state['thread_id'])

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_thread_ids'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conv(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['history'] = temp_messages

for message in st.session_state['history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('chat with me')

if user_input:
    st.session_state['history'].append({'role':'user','content':user_input})
    with st.chat_message("user"):
        st.text(user_input)
    
    CONFIG = {"configurable":{'thread_id':st.session_state['thread_id']},"metadata":{"thread_id":st.session_state["thread_id"]},"run_name":"chat_turn"}

    with st.chat_message("assistant"):
        status_holder = {"box":None}
        def ai_stream():
            for message_chunk, metadata in agent.stream({"messages":[HumanMessage(content=user_input)]},config=CONFIG,stream_mode="messages"):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state['history'].append({'role':'assistant','content':ai_message})