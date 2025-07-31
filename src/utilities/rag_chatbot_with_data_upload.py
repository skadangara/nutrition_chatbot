import warnings
import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from libs import preprocess_data as pr

# --- Setup ---
warnings.filterwarnings("ignore")
load_dotenv()


# --- Helpers ---
def build_documents_from_faq(faq_data):
    docs = []
    for item in faq_data:
        for q in item["questions"]:
            content = q.strip() + "\n\n" + item["answer"].strip()
            docs.append(Document(page_content=content))
    return docs


def index_docs(documents, persist_path="./chroma_db"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_path,
    )
    return vectorstore


def get_retriever(vectorstore, search_type="similarity_score_threshold", similarity_threshold=0.7, top_k=1):
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"score_threshold": similarity_threshold, "k": top_k}
    )


def get_generator(model="gpt-4o", temperature=0, max_tokens=None, timeout=None, max_retries=2):
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate_response(message, retriever, llm):
    ret_docs = retriever.invoke(message)

    if len(ret_docs) == 0:
        return "Sorry, I don’t have information on that. Please try a different question."

    rag_prompt = hub.pull("rlm/rag-prompt")

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    return qa_chain.invoke(message)

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Diabetes Nutrition Chatbot", page_icon="🥗")
    st.title("🥗 Diabetes Nutrition Chatbot")
    st.markdown("Ask me any nutrition-related question! You can upload your own FAQ file.")

    # Session State Initialization
    if "vectorstore" not in st.session_state:
        # Load default FAQ
        default_docs = pr.load_data("data/faq_data.json")
        st.session_state.vectorstore = index_docs(default_docs, persist_path="./chroma_db")
        st.session_state.retriever = get_retriever(st.session_state.vectorstore)
        st.session_state.llm = get_generator()
        st.session_state.messages = []
        st.session_state.faq_status = "✅ Default FAQ loaded."

    # FAQ Upload UI
    uploaded_file = st.file_uploader("Upload a custom FAQ JSON", type="json")
    if uploaded_file is not None:
        try:
            faq_json = json.load(uploaded_file)
            uploaded_docs = build_documents_from_faq(faq_json)
            st.session_state.vectorstore = index_docs(uploaded_docs, persist_path="./chroma_uploaded_db")
            st.session_state.retriever = get_retriever(st.session_state.vectorstore)
            st.session_state.faq_status = "✅ Custom FAQ uploaded and indexed."
        except Exception as e:
            st.session_state.faq_status = f"❌ Failed to load FAQ: {e}"

    # Show current FAQ status
    st.info(st.session_state.faq_status)

    # Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask me any nutrition-related question!"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            reply = generate_response(prompt, st.session_state.retriever, st.session_state.llm)
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
