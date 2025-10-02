import streamlit as st
from project_4 import agent,retrive_all_threads
from langchain_core.messages import HumanMessage,AIMessageChunk,ToolMessageChunk
import requests
import json

api_url = "http://127.0.0.1:8000/"

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_thread_ids"]:
        st.session_state["chat_thread_ids"].append(thread_id)

def reset_chat():
    id = requests.get(api_url + "uuid").json()
    print(id)
    st.session_state['thread_id'] = id["uuid"]
    add_thread(st.session_state['thread_id'])
    st.session_state['history'] = []

def load_conv(thread_id):
    state = agent.get_state(config={'configurable':{"thread_id":thread_id}})
    return state.values.get('messages',[])

if "history" not in st.session_state:
    st.session_state['history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = requests.get(api_url + "uuid").json()['uuid']

if 'chat_thread_ids' not in st.session_state:
    st.session_state['chat_thread_ids'] = requests.get(api_url + "get_all_threads").json()['threads']

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
            stream_chunk = requests.post(url= api_url + "prompt",json={"thread_id":st.session_state["thread_id"],"input":user_input,"mode":"stream"},stream=True)
            print(stream_chunk)
            for line in stream_chunk.iter_lines():
                message_chunk = json.loads(line)
                print(message_chunk)
                if message_chunk[0]['id'][3] == 'ToolMessage':
                    tool_name = message_chunk[0]['kwargs']['name']
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

                if message_chunk[0]['kwargs']['type'] == 'AIMessageChunk':
                    yield message_chunk[0]['kwargs']['content']

        ai_message = st.write_stream(ai_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state['history'].append({'role':'assistant','content':ai_message})