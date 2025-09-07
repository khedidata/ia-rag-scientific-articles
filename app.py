# app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pathlib

# --- importe ton RAG tel que tu l'as défini ---
# Assure-toi que PYTHONPATH pointe vers ton projet ou adapte les imports:
from chains.conversational_qa import rag_chain

app = FastAPI(title="RAG API")

# CORS (utile si tu sers le front ailleurs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restreins en prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    message: str

# Sert le front (index.html) depuis disque
@app.get("/", response_class=HTMLResponse)
def index():
    html = pathlib.Path("index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)

@app.post("/chat")
def chat(inp: ChatInput):
    user_msg = inp.message.strip()
    if not user_msg:
        return JSONResponse({"answer": "Please write a message."}, status_code=400)
    # Appel à ta chaîne RAG (retourne un dict avec clé "answer" selon ton wiring)
    result = rag_chain.invoke(user_msg)
    # Si jamais ta chaîne retourne juste un str, ajuste ici:
    answer = result["answer"] if isinstance(result, dict) and "answer" in result else str(result)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


