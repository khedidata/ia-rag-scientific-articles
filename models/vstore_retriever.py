from langchain.vectorstores import FAISS, VectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from constants import EMBEDDINGS_MODEL_NAME, FAISS_INDEX_PATH


embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)

vector_store: VectorStore = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
    )

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" : 5,
                     "return_score" : True}
    )