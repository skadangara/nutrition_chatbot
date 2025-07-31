
# Project Title
Nutrition Chatbot for People with
Diabetes







## Project Summary

This is a chatbot that answers nutrition-related questions for people with diabetes using a provided FAQ dataset.

## Approaches
Approaches:

Here mainly 2 approaches implemented.

Approach1: 

In the first approach a NLP chat bot is implemented using huggingface sentence transformers and semantic similarity. Streamlit UI is integrated with the Bot for helping the users to interact with the bot in realtime. A baseline threshold of 0.7 is used to fetch the most relevant answers for the user query, if the fetched docs have less similarity than the threshold, a fallback message will be displayed for the user query.

Approach2:

In the second approach,  it implements a Retrieval-Augmented Generation (RAG) chatbot designed to interact with FAQ nutrition dataset. It leverages state-of-the-art language models GPT-4o for generation and all-MiniLM-L6-v2 for embeddings. Built with LangChain and served via a Streanlit interface, the chatbot provides an interactive platform to query any nutrition related questions without relying on cloud-based services. The vector storage is handled using ChromaDB, ensuring efficient semantic retrieval of document content. A baseline threshold of 0.7 is used to fetch the most relevant answers for the user query, if the fetched docs have less similarity than the threshold, a fallback message will be displayed for the user query.
## Prerequisites

1. Unzip the code repo
2. Set LangChain API KEY and OpenAI API KEY in .env file.
3. Install the dependencies from the requirements file.
4. Make sure to keep the faq json file inside the data folder in the code repo.
## Execution

Run the Chatbot:

1. Open a terminal and cd to the code repo.
2. To run the first approach bot, use the below command

streamlit run src/nlp_bot.py

3. To run the second approach bot, use the below command.

4. After successful run, a link  http://localhost:8501 will be opened in your default browser.

streamlit run src/llm_rag_chatbot.py
## Extended Implementations

Further Implementations:

The below 2 extended versions also implemented which are kept inside the utilities folder. To run this versions, first copy the below files to src folder and run as below.

"llm_rag_chatbot_top_k_answers" : This method uses RAG-llm method which displays top 2 answers with its similarity scores. To run this use below command after copying it to src folder.

streamlit run src/llm_rag_chatbot_top_k_answers.py

"rag_chatbot_with_data_upload": This method uses RAG-llm method with an option to upload custom dataset which can be used for answering the questions. To run this

streamlit run src/rag_chatbot_with_data_upload.py
## Tools & Technologies

LLM : GPT-4o
Embedding : all-MiniLM-L6-v2 huggingface
Framework : LangChain
Programming Language : Python
Vector Store : ChromaDB
User Interface : Streamlit

## Documentation

Refer the "docs/usecase_tech_doc.pdf" document  for the details of the overall achitecture, similarity threshold strategy and the chunking methods.


## Authors

@skadangara
## License

[Sajana Kadangara]
