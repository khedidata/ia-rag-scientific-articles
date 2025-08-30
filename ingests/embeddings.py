import torch
from langchain_huggingface import HuggingFaceEmbeddings

from constants import EMBEDDINGS_MODEL_NAME

def get_embeddings_model(model: str = EMBEDDINGS_MODEL_NAME):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name=model, 
        model_kwargs={"device" : device}  
        )