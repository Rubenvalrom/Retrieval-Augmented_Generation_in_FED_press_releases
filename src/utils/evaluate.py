from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import mlflow
from operator import itemgetter

from .prompts import get_judge_1_prompt, get_judge_2_prompt, get_judge_3_prompt
from .format import parse_with_fixer


def log_params_from_collection_name(name: str):
    """
    Extracts parameters from the collection name and logs them to MLflow.
    if the collection name starts with "Recursive", it extracts chunk size and overlap.
    if it starts with "Semantic", it extracts the percentile.
    """
    if name.startswith("Recursive"):
        split_method = "Recursive"
        chunk_size = name[25:29]
        if "_" in chunk_size:
            chunk_size = chunk_size.replace("_", "")
        overlap = name[-2:]
        overlap = int(overlap)
        chunk_size = int(chunk_size)
        mlflow.log_param("split_method", split_method)
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("overlap", overlap)

        print(f"Logged Recursive params: chunk_size={chunk_size}, overlap={overlap}")
        print('-'*50)

        return 
    
    elif name.startswith("Semantic"):
        split_method = "Semantic"
        percentile = name[17:21]
        if "th" in percentile:
            percentile = percentile[0:2]
        percentile = float(percentile)

        mlflow.log_param("split_method", split_method)
        mlflow.log_param("percentile", percentile)

        print(f"Logged Semantic params: percentile={percentile}")
        print('-'*50)

        return
    else:
        print("Unknown collection name format for logging parameters.")
        print('-'*50)

        return 

def evaluate_query(answer, llm, query_id):
    if query_id == 1:
        system_prompt = get_judge_1_prompt()
    elif query_id == 2:
        system_prompt = get_judge_2_prompt()
    elif query_id == 3:
        system_prompt = get_judge_3_prompt()

    rag_chain = (
        {  
            "generated_answer": itemgetter("generated_answer")
        }
        | system_prompt
        | llm.bind(stop=["Human:", "System:"])
        | StrOutputParser()
        | RunnableLambda(parse_with_fixer)
        )

    results = rag_chain.invoke({
            "generated_answer": answer
        })
    
    score = 0
    for key, value in results.items():
        # Each boolean criterion adds 1 to the score if true
        # Since it might be a string instead of a boolean, it is converted to string
        bool_value = 1 if str(value).lower() == "true" else 0
        score += bool_value
        
        # Log each criterion result separately
        mlflow.log_metric(f"Q{query_id}_{key}", bool_value)

    # Log the final score for the query
    mlflow.log_metric(f"Q{query_id}_final_score", score)
    return results, score

def log_params_from_collection_name(name: str):
    """
    Extracts parameters from the collection name and logs them to MLflow.
    if the collection name starts with "Recursive", it extracts chunk size and overlap.
    if it starts with "Semantic", it extracts the percentile.
    """
    if name.startswith("Recursive"):
        split_method = "Recursive"
        chunk_size = name[25:29]
        if "_" in chunk_size:
            chunk_size = chunk_size.replace("_", "")
        overlap = name[-2:]
        overlap = int(overlap)
        chunk_size = int(chunk_size)
        mlflow.log_param("split_method", split_method)
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("overlap", overlap)

        print(f"Logged Recursive params: chunk_size={chunk_size}, overlap={overlap}")
        print('-'*50)

        return 