
import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


# Step 1: Initialize LLM
llm = ChatGroq(
    model=os.getenv("GROQ_MODEL_NAME"),
    temperature=0.5,
    max_tokens=512,
    api_key=os.getenv("GROQ_API_KEY"),
)


# Step 2: Load FAISS Database
DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)


# Step 3: Create Custom Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer the user's question using the given context.
    If the answer is not available in the context, say you don't know.

    Context:
    {context}"""),

    ("human", "{input}")
])


# Step 4: Create Document Chain
combine_docs_chain = create_stuff_documents_chain(
    llm,
    prompt
)


# Step 5: Create Retrieval Chain
rag_chain = create_retrieval_chain(
    db.as_retriever(search_kwargs={"k": 3}),
    combine_docs_chain
)


# Step 6: Ask Question
user_query = input("Write Query Here: ")

response = rag_chain.invoke({"input": user_query})

print("\nRESULT:")
print(response["answer"])


# Step 7: Display Source Documents
print("\nSOURCE DOCUMENTS:")

for doc in response["context"]:
    print(f"\nMetadata: {doc.metadata}")
    print(f"Content: {doc.page_content[:200]}...")
