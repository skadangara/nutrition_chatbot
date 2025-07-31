import gradio as gr
import streamlit as st
import json
import re
import torch
from sentence_transformers import SentenceTransformer, util
from libs import preprocess_data as pr

# --- Load model and FAQ ---
@st.cache_resource
def load_model_and_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # load data and preprocess
    questions, answers = pr.load_flatten_data("data/faq_data.json")

    embeddings = model.encode(questions, convert_to_tensor=True)
    return model, questions, answers, embeddings

# Initialise model, questions , answers and embeddings
model, questions, answers, question_embeddings = load_model_and_data()

# --- Bot logic ---
def get_response(query, threshold=0.7):
    query_proc = pr.preprocess(query)
    query_embedding = model.encode(query_proc, convert_to_tensor=True)

    cosine_scores = util.pytorch_cos_sim(query_embedding, question_embeddings)

    best_score = torch.max(cosine_scores).item()
    best_idx = torch.argmax(cosine_scores).item()

    if best_score >= threshold:
        return answers[best_idx]
    else:
        return "Sorry, I don’t have information on that. Please try a different question."

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

    if prompt := st.chat_input("Ask me any nutrition-related question!"):
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            response = get_response(prompt)
            st.markdown(f"**Answer:** {response}")
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()




