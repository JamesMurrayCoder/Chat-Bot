import streamlit as st
from utils import write_message
from agent import generate_response

# Page Config
st.set_page_config("Chat Bot", page_icon=":robot_face:")

# Set up Session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"},
    ]


# Submit handler
def handle_submit(message):

    # Handle the response
    with st.spinner('Thinking...'):
        # Call to the LLM
        response = generate_response(message)
        write_message('assistant', response)


# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save = False)

# Handle user input
if question := st.chat_input("How can I help you?"):
    # Display message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(question)
