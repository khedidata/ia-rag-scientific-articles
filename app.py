from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pathlib

from chains.conversational_qa import rag_chain


app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    message: str


@app.get("/", response_class=HTMLResponse)
def index():
    html = pathlib.Path("index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)

@app.post("/chat")
def chat(inp: ChatInput):
    user_msg = inp.message.strip()
    if not user_msg:
        return JSONResponse({"answer": "Please write a message."}, status_code=400)
    result = rag_chain.invoke(user_msg)
    answer = result["answer"] if isinstance(result, dict) and "answer" in result else str(result)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


