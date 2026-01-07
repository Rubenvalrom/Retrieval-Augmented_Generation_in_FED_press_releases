from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb import HttpClient
import joblib
import torch

# Initial setup
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
    print("Chroma server is not running. Please start the server and try again.")
    raise e

print('-'*50)

# Load cleaned documents
documents = joblib.load('./clean/clean_documents.pkl')
print(f"Number of documents loaded: {len(documents)}")
print('-'*50)

# Initialize embedding models

# Model for semantic chunking (smaller model)
chunk_embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

# Model for final embeddings (larger model)
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)

def semantic_chunk(documents, threshold, chunk_embedding_model):
    """
    Chunk documents using SemanticChunker based on the given threshold.
    Args:
        documents (list): List of documents to be chunked.
        threshold (float): Percentile threshold for chunking.
        chunk_embedding_model: Embedding model used for chunking.
    Returns:
        chunks (list): List of chunked documents.
        collection_name (str): Name for the Chroma collection.
    """
    text_splitter = SemanticChunker(
        chunk_embedding_model,
        breakpoint_threshold_type = "percentile", 
        breakpoint_threshold_amount = threshold,
        min_chunk_size = 100,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")
    print('-'*50)

    collection_name = f"Semantic_chunker_{threshold}th_percentile"

    return chunks, collection_name

def recursive_chunk(documents, chunk_size, chunk_overlap_percentage):
    """
    Chunk documents using RecursiveCharacterTextSplitter.
    Args:
        documents (list): List of documents to be chunked.
        chunk_size (int): Size of each chunk.
        chunk_overlap_percentage (int): Overlap percentage between chunks.
    Returns:
        chunks (list): List of chunked documents.
        collection_name (str): Name for the Chroma collection.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = (chunk_size * chunk_overlap_percentage) // 100,
        length_function = len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")
    print('-'*50)

    collection_name = f"Recursive_character_size-{chunk_size}_overlap-{chunk_overlap_percentage}"

    return chunks, collection_name

def insert_data_to_chroma(chunks, collection_name, embedding_model, client):
    """
    Insert chunked documents into Chroma collection in batches.
    Args:
        chunks (list): List of chunked documents.
        collection_name (str): Name for the Chroma collection.
        embedding_model: Embedding model used for creating embeddings.
        client: Chroma client instance.
    """
    collections = client.list_collections()
    collections = [col.name for col in collections]

    if collection_name in collections:
        print(f"Collection '{collection_name}' already exists.")
        return
    
    vectorstore = Chroma(client=client, 
                     collection_name=collection_name, 
                     embedding_function=embedding_model)
    
    # The limit of chunks inserted at the same time is 5461
    batch_size = 5460  

    # Insert chunks in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)

    print(f"{collection_name} has been indexed")
    
# Percentile thresholds for semantic chunking
thresholds = [50, 75, 90, 97.5] 

# For each threshold, chunk documents and insert into Chroma
for threshold in thresholds:
    chunks, collection_name = semantic_chunk(documents, threshold, chunk_embedding_model)
    insert_data_to_chroma(chunks, collection_name, embedding_model, client)
    print('='*50)

# Parameters for recursive character chunking
chunk_sizes = [500, 1000, 1500]
overlap_percentages = [10, 15]

# For each combination of chunk size and overlap percentage, chunk documents and insert into Chroma
for chunk_size in chunk_sizes:
    for overlap_percentage in overlap_percentages:
        chunks, collection_name = recursive_chunk(documents, chunk_size, overlap_percentage)
        insert_data_to_chroma(chunks, collection_name, embedding_model, client)
        print('='*50)

# List all collections in Chroma and their size after insertion
print("Collections completed:")
print("-"*51)
for i, col in enumerate(client.list_collections()):
    print(f" {i+1}. {col.name} - {col.count()}")
print("-"*51)
