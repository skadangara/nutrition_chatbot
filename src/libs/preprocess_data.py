import re
import json
from langchain.schema import Document


# --- Preprocessing ---
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def load_data(filename):
    # Load FAQ data
    with open(filename) as f:
        faq_data = json.load(f)
    # Flatten into individual Documents
    documents = []
    for item in faq_data:
        for q in item["questions"]:
            doc = Document(page_content=preprocess(q) + "\n\n" + preprocess(item["answer"]))
            documents.append(doc)

    return documents

def load_flatten_data(filename):
    with open(filename) as f:
        faq_data = json.load(f)

    questions = []
    answers = []

    for faq in faq_data:
        for q in faq["questions"]:
            questions.append(preprocess(q))
            answers.append(faq["answer"])

    return questions, answers