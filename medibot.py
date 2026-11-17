
import os
import streamlit as st

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


# Step 1: Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# Step 2: Create Custom Prompt
def set_custom_prompt():
    custom_prompt = """
    Use the following context to answer the user's question.
    If you don't know the answer from the context, say that you don't know.

    Context:
    {context}

    Question:
    {input}

    Answer:
    """

    prompt = PromptTemplate(
        template=custom_prompt,
        input_variables=["context", "input"]
    )

    return prompt


# Step 3: Main Streamlit Application
def main():

    st.title("Ask Medibot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:

        st.chat_message("user").markdown(prompt)

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        try:

            # Load Vector Database
            vectorstore = get_vectorstore()

            # Initialize Groq LLM
            llm = ChatGroq(
                model=os.getenv("GROQ_MODEL_NAME"),
                temperature=0.5,
                max_tokens=512,
                api_key=os.getenv("GROQ_API_KEY")
            )

            # Create Custom Prompt
            custom_prompt = set_custom_prompt()

            # Create Document Chain
            combine_docs_chain = create_stuff_documents_chain(
                llm,
                custom_prompt
            )

            # Create Retrieval Chain
            rag_chain = create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={"k": 3}),
                combine_docs_chain
            )

            # Generate Response
            response = rag_chain.invoke({"input": prompt})

            result = response["answer"]

            # Display Answer
            st.chat_message("assistant").markdown(result)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
