import warnings
import os
warnings.filterwarnings("ignore")
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import json
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from libs import preprocess_data as pr

# Setting langchain key
load_dotenv()

# Load FAQ data

documents = pr.load_data("data/faq_data.json")

def index_docs(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="./chroma_db",
    )
    return vectorstore

def get_retriever(vectorstore, search_type, similarity_threshold, top_k):
    retriever = vectorstore.as_retriever(search_type=search_type,
                                         search_kwargs={"score_threshold": similarity_threshold, "k": top_k})
    return retriever


def get_generator(model="gpt-4o",temperature=0,max_tokens=None,timeout=None,max_retries=2):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries
    )
    return llm


# Embed and index documents
vectorstore = index_docs(documents)
# define retriever
retriever = get_retriever(vectorstore,"similarity_score_threshold",0.7,2)
# Initialising LLM model for generator purpose
llm = get_generator()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def response(message):

    ret_docs =  retriever.invoke(message)

    top_docs = retriever.vectorstore.similarity_search_with_score(message, k=2)
    suggestions = []
    for doc, score in top_docs:
        question, answer = doc.page_content.strip().split("\n\n", maxsplit=1)
        suggestions.append((answer.strip(), round(score, 2)))

    if len(ret_docs) == 0:
        return "Sorry, I don’t have information on that. Please try a different question."

    # getting prompt template from langchain hub
    rag_prompt = hub.pull("rlm/rag-prompt")
    # Initialising langchain QA chain with Retriever
    qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
    )
    result = qa_chain.invoke(message)
    return {"answer":result, "suggestions":suggestions}

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Diabetes Nutrition Chatbot", page_icon="🥗")
    st.title("🥗 Diabetes Nutrition Chatbot")
    st.markdown("Ask me any nutrition-related question!")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


if __name__ == "__main__":
    main()
    if prompt := st.chat_input("Ask me any nutrition-related question!"):
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            response = response(prompt)
        #     st.markdown(response)
        # st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response["answer"])
            if response["suggestions"]:
                with st.expander("🔍 See similar answers and scores"):
                    for i, (answer, score) in enumerate(response["suggestions"]):
                        st.markdown(f"""
                    **Suggestion {i + 1}:**
                    - **Answer:** {answer}
                    - **Score:** {score}
                    """)

        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
