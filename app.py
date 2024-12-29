import streamlit as st
from langchain.memory import StreamlitChatMessageHistory
from llm_chains import load_normal_chain, load_pdf_chat_chain
from pdf_handler import add_documents_to_db
from utils import save_chat_history_json, load_chat_history_json, get_timestamp
import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_chain(chat_history):
    if st.session_state.pdf_chat:
        print("DEBUG: Loading pdfChatChain...")
        return load_pdf_chat_chain(chat_history)
    else:
        print("DEBUG: Loading chatChain...")
        return load_normal_chain(chat_history)

def clear_input_field():
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    st.session_state.send_input = True
    clear_input_field()

def track_index():
    st.session_state.session_index_tracker = st.session_state.session_key

def save_chat_history():
    if st.session_state.history != []:
        if st.session_state.session_key == "sesiune_noua":
            st.session_state.new_session_key = get_timestamp() + ".json"
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.new_session_key)
        else:
            save_chat_history_json(st.session_state.history, config["chat_history_path"] + st.session_state.session_key)

def toggle_pdf_chat():
    st.session_state.pdf_chat = True

def main():
    st.title("Aplicatie RAG")
    chat_container = st.container()
    st.sidebar.title("Sesiuni de conversatie")
    chat_sessions = ["sesiune_noua"] + os.listdir(config["chat_history_path"])

    if "send_input" not in st.session_state:
        st.session_state.session_key = "sesiune_noua"
        st.session_state.send_input = False
        st.session_state.user_question = ""
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "sesiune_noua"

    if st.session_state.session_key == "sesiune_noua" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    if st.session_state.session_key != "sesiune_noua":
        st.session_state.history = load_chat_history_json(config['chat_history_path'] + st.session_state.session_key)
    else:
        st.session_state.history = []

    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Alegeti sesiunea de conversatie", chat_sessions, key="session_key", index=index, on_change=track_index)
    st.sidebar.toggle("PDF Chat", key="pdf_chat", value=False, on_change=toggle_pdf_chat)
    
    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    user_input = st.text_input("Introduceti mesajul aici", key="user_input", on_change=set_send_input)
    send_button = st.button("Trimite", key="send_button")

    uploaded_file = st.sidebar.file_uploader("Incarca un fisier PDF", accept_multiple_files=True, key="pdf_upload", type=["pdf"], on_change=toggle_pdf_chat)
    if uploaded_file:
        with st.spinner("Se incarca fisierul..."):
            add_documents_to_db(uploaded_file)

    if send_button or st.session_state.send_input:
        if st.session_state.user_question != "":
            with chat_container:
                st.chat_message("user").write(st.session_state.user_question)

                # Format chat history into tuples
                formatted_chat_history = [
                    (message.type, message.content) for message in chat_history.messages
                ]
                print(st.session_state.pdf_chat)
                # Adjust input keys based on chat mode
                if st.session_state.pdf_chat:
                    print("dadaadadsdsadas")
                    inputs = {
                        "question": st.session_state.user_question,
                        "chat_history": formatted_chat_history,
                    }
                else:
                    print("nububununnunu")
                    inputs = {
                        "human_input": st.session_state.user_question,
                        "chat_history": formatted_chat_history,
                    }

                # Debug inputs
                print(f"Inputs passed to llm_chain: {inputs}")

                # Run the appropriate chain
                llm_response = llm_chain.run(inputs)

                st.chat_message("ai").write(llm_response)
                st.session_state.user_question = ""

    if chat_history.messages != []:
        with chat_container:
            st.write("Istoricul conversatiei:")
            for message in chat_history.messages:
                st.chat_message(message.type).write(message.content)

    save_chat_history()

if __name__ == "__main__":
    main()