from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma
from chromadb import HttpClient
from operator import itemgetter
import torch

from utils.llms import load_model, load_embedding_model
from utils.format import parse_with_fixer, format_docs
from utils.prompts import get_system_prompt

def rag(query):

    # Initial Set up
    print('-'*50)
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print('-'*50)

    # Check if Chroma server is running and connect to it
    try:
        client = HttpClient(host="http://localhost:8000")
        print("Chroma server is running.")
    except Exception as e:
        raise e

    print('-'*50)

    # Load embedding model
    embedding_model = load_embedding_model(device=device)

    print("Embedding model loaded successfully.")
    print('-'*50)

    # Load llm model and his system prompt
    prompt = get_system_prompt()
    llm = load_model()

    # Best retrieval parameters from experiments
    collection_name = "Recursive_character_size-1500_overlap-15"
    k = 20

    try:
        # Load Chroma collection
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            client=client
        )
    except Exception as e:
        print(f"Failed to load Chroma collection: {collection_name}")
        raise e
    
    print(f"Chroma collection '{collection_name}' loaded successfully.")
    print('-'*50)

    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
    "k": k,
    "fetch_k": k * 5,
    "lambda_mult": 0.7             
        }
    )

    rag_chain = (
    {
        # Extract the string first before giving it to the retriever
        "context": (itemgetter("question")) | retriever | format_docs, 
        "question": itemgetter("question")
    }
    | prompt
    | llm.bind(stop=["Human:", "System:"])
    | StrOutputParser()
    | RunnableLambda(parse_with_fixer)
    )

    answer = rag_chain.invoke({"question": query})

    return answer

if __name__ == "__main__":
    question = "Provide a sentiment analysis of the early 2024 Federal Reserve press releases and justify your answer with specific references to the text."
    answer = rag(question)   
    print(answer)
    input("Press Enter to continue...")