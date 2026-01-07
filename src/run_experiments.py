import json
import time
import torch
import mlflow
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb import HttpClient
from operator import itemgetter

from utils.prompts import get_system_prompt
from utils.evaluate import log_params_from_collection_name, evaluate_query, parse_with_fixer
from utils.llms import load_model, load_judge_model

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
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

print("Embedding model loaded successfully.")
print('-'*50)

# Conect to Mlflow Tracking Server
try:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Fed_Press_Conferences_Analysis")
    print("Connected to MLflow Tracking Server.")
    print('-'*50)
    print("Experiment: Fed_Press_Conferences_Analysis")
    print('-'*50)

except Exception as e:
    print("Failed to connect to MLflow Tracking Server.")
    raise e


# Format documents for rag chain context
def format_docs(docs):
    formatted = []
    # Iterate through documents, extract metadata and content
    for doc in docs:
        # Extract metadata
        meta = doc.metadata
        date = meta.get('creationdate', 'Unknown Date')
        page = meta.get('page', '?')
        total_pages = meta.get('total_pages', '?')

        # Clean content by replacing newlines with spaces
        content = doc.page_content.replace("\n", " ")

        formatted.append(f"FRAGMENT [Date: {date} | Page: {page} of {total_pages}] \n{content}")
    
    # Combine all formatted documents into a single context string separated by double newlines
    context = "\n\n".join(formatted)

    return context

prompt = get_system_prompt()

llm = load_model()
llm_judge = load_judge_model()

TEST_QUERIES = [
    # 1. Covid evolution (2021)
    "How did the sentiment and usage of the term 'transitory' to describe inflation evolve in press conferences throughout 2021? When did the tone shift from confident to concerned?",
    
    # 2. Crisis comparison (2008 vs 2020)
    "Compare the tone of urgency regarding unemployment post-2008 versus the tone during the onset of the pandemic in 2020.",
    
    # 3. Specific Fact Retrieval (2025)
    "What was the specific interest rate decision announced in the December 2025 press conference, and how did Chair Powell describe the availability of federal government data regarding the economic outlook?"
]

def run_experiment(collection_name: str, k_mmr: int):

    # Define run name
    run_name = f"{collection_name}_k-{k_mmr}"
    print(f"Running experiment: {run_name}")
    print('-'*50)
    mlflow.start_run(run_name=run_name)
    log_params_from_collection_name(collection_name)
    mlflow.log_param("k", k_mmr)
    
    # Load Chroma collection
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        client=client
    )
    
    # MMR Retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
        "k": k_mmr,
        "fetch_k": k_mmr * 5,
        "lambda_mult": 0.7 # Higher lambda favors relevance over diversity
                           # Initially was going to test different values, but token limit constraints
        }
    )
    
    # Rag chain
    rag_chain = (
    {
        # Extract the string first, THEN pipe it to the retriever
        "context": (itemgetter("question")) | retriever | format_docs, 
        "question": itemgetter("question")
    }
    | prompt
    | llm.bind(stop=["Human:", "System:"])
    | StrOutputParser()
    | RunnableLambda(parse_with_fixer)
    )

    overall_score = 0
    for query_id, query in enumerate(TEST_QUERIES, start=1):
        print(f"Processing query: {query_id}")
        print('-'*50)

        # Run the RAG chain
        start = time.time()
        answer = rag_chain.invoke({"question": query})
        mlflow.log_text(json.dumps(answer, indent=2), f"answer_query_{query_id}.json")  

        results, score = evaluate_query(answer, llm_judge, query_id)
        overall_score += score
        print(f"Results for query {query_id}: {json.dumps(results, indent=2)}")
        print('-'*50)

        duration = time.time() - start
        if duration < 61:
            sleep_time = 61 - duration
            time.sleep(sleep_time)

    mlflow.log_metric("overall_score", overall_score)
    mlflow.end_run()

    print(f"Overall score for experiment {run_name}: {overall_score}")

    return 

# Experiment execution loop

# Load Chroma collections
try:
    collections = client.list_collections()
    print("Collections:")
    for i, col in enumerate(collections):
        print(f" {i+1}. {col.name} - {col.count()}")
    print("-"*51)

    collections = [col.name for col in collections]
except Exception as e:
    print("Failed to retrieve collections from Chroma server.")
    raise e

# Number of documents to retrieve
k_values = [15, 30, 45]

for collection_name in collections[0:2]:
    for k in k_values:
        run_experiment(collection_name, k)

input("All experiments completed. Press Enter to exit.")
