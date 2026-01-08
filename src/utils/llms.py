from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import torch

from dotenv import load_dotenv

load_dotenv()

def load_model():
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.25,
        max_tokens=None,
        timeout=None,
        max_retries=3,
        )   
    return llm

def load_judge_model():
    # Configure 4-bit quantization, the 16-bit version does not fit in my gpu (12GB)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        # Gemma was trained with bfloat16: 1 sign bit, 8 exponent bits, 7 mantissa bits
        bnb_4bit_compute_dtype=torch.bfloat16,
        # Scales reduced from 16 bit to 4 bit too
        bnb_4bit_use_double_quant=True,
    )

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.001,
        return_full_text=False
    )

    # Wrap it in a LangChain-compatible object
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

def load_embedding_model(device="cpu"):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embedding_model