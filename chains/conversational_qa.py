from typing import List

from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.schema import Document
from langchain_core.runnables.base import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

from constants import LLM_MODEL_NAME
from models import vector_store
from prompt import ANSWER_PROMPT, REPHRASE_PROMPT
from utils import format_context, reciprocal_rank_fusion
from config import OPENAI_API_KEY



# ========== LLM (LARGE LANGUAGE MODEL) ========== # 
llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, api_key=OPENAI_API_KEY)


# ========== Conversation Memory ========== #
memory = ConversationBufferWindowMemory(
    memory_key = "chat_history",
    return_messages = True,
    k=10
)


# ======================================================
#    RAG (Retrieval Augmented Generation) Chain Steps
# ======================================================

def generate_alternative_queries(query: str,
                                 n: int = 5) -> List[str]:
    """
    Generate a list of alternative phrasings for a given query using the LLM.

    This function sends the original query along with the chat history to the
    language model (LLM) to produce multiple rephrased versions of the query.
    It ensures uniqueness by ignoring case and returns the original query
    followed by up to `n` unique alternative queries.

    Args:
        query (str): The original user query.
        n (int, optional): The maximum number of alternative queries to generate.
                           Defaults to 5.

    Returns:
        List[str]: A list containing the original query and its unique alternative
                   phrasings, limited to `n + 1` items in total.
    """
    messages = REPHRASE_PROMPT.format_messages(
        question=query,
        n=n,
        chat_history=memory.chat_memory.messages
    )
    responses = llm.invoke(messages)
    lines = [q.strip() for q in responses.content.splitlines() if q.strip()]
    queries = [query] + lines
    seen, unique = set(), []
    for q in queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)
            continue
    return unique[: n + 1]

decompose_chain = RunnableLambda(generate_alternative_queries)

def retrieval_and_fusion(queries: List[str],
                        k_per_query: int = 5,
                        rrf_k: int = 60,
                        top_n: int = 5
                    ) -> List[Document]:
    """
    Retrieve documents for multiple queries and combine the results using Reciprocal Rank Fusion (RRF).

    For each query in `queries`, this function retrieves the top `k_per_query` documents
    from the vector store. Then it applies Reciprocal Rank Fusion to merge the results
    across all queries, producing a single ranked list of documents. Finally, only the
    Document objects are returned, discarding their scores.

    Args:
        queries (List[str]): A list of query strings to search for.
        k_per_query (int, optional): Number of top documents to retrieve per query. Defaults to 5.
        rrf_k (int, optional): The k parameter for Reciprocal Rank Fusion. Defaults to 60.
        top_n (int, optional): Number of top documents to return after fusion. Defaults to 5.

    Returns:
        List[Document]: A list of fused Document objects, ranked according to RRF.
    """
    per_query_results = [vector_store.similarity_search(q, k=k_per_query) for q in queries]
    fused = reciprocal_rank_fusion(per_query_results, k=rrf_k, top_n=top_n)
    fused_docs = [doc for doc, _ in fused]
    return fused_docs


retrieval_chain = RunnableLambda(retrieval_and_fusion)

def generate_answer(docs: List[Document], query: str) -> str:
    """
    Generate an answer to a given query using a list of retrieved documents and update chat memory.

    This function formats the content of the provided documents into a context string,
    then constructs messages using the `ANSWER_PROMPT`. It invokes the LLM to generate
    an answer, and updates the conversation memory with both the user query and the
    AI's response.

    Args:
        docs (List[Document]): A list of Document objects containing relevant information for the query.
        query (str): The user's question to answer.

    Returns:
        str: The content of the AI-generated (LLM) answer.
    """
    chat_history = memory.chat_memory.messages if memory.chat_memory.messages else []

    context_str = format_context(docs)
    final_messages = ANSWER_PROMPT.format_messages(
        chat_history=chat_history,
        context=context_str,
        question=query
    )
    answer_messages = llm.invoke(final_messages)
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(answer_messages.content)
    return answer_messages.content


# =====================
#    Final RAG Chain
# =====================

rag_chain = (
    # Entry point: the input x is the user question
    RunnableLambda(lambda x: {"question": x, "history": "", "use_memory": bool(memory.chat_memory.messages)})  
    | {
        "question": lambda x: x["question"],
        # 1. Generate alternative queries from the original question
        "queries": lambda x: decompose_chain.invoke(x["question"]),
        # Keep the conversation history
        "history": lambda x: x.get("history", "")
    }
    | {
        "question": lambda x: x["question"],
        # 2. Retrieve and fuse relevant documents for all alternative queries
        "docs": lambda x: retrieval_chain.invoke(x["queries"]),
        # Preserve history
        "history": lambda x: x.get("history", "")
    }
    | {
        "question": lambda x: x["question"],
        # 3. Generate the final answer from the retrieved documents
        "answer": lambda x: generate_answer(x["docs"], x["question"]),
        # Preserve history
        "history": lambda x: x.get("history", "")
    }
)