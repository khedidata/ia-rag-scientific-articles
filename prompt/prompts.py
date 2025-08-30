from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder


# Prompt to generate alternative phrasings of the user's question
REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    # System message instructing the LLM on how to rephrase the question
    ("system",
     "You rewrite the user's question into {n} semantically different queries "
     "about scientific computer science (arXiv) abstracts, short and specific. "
     "Return one query per line, no bullets, no numbering."),
    # Placeholder to include previous chat messages for context
    MessagesPlaceholder("chat_history"),
    # User's original question to be rephrased
    ("user", "{question}")
])

# Prompt to generate the final answer based on retrieved documents
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    # System message instructing the LLM to be helpful and only use the provided context
    ("system",
     "You are a helpful AI assistant for scientific Q&A on arXiv CS abstracts. "
     "Use ONLY the provided context to answer. If unsure, say you don't know."),
    # Include previous chat history to maintain conversation context
    MessagesPlaceholder("chat_history"),
    # System message containing the context retrieved from documents
    ("system", "Context:\n{context}"),
    # User's current question to answer
    ("user", "{question}")
])