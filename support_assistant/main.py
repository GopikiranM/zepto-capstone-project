from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask, ensure_index, Answer

app = FastAPI(title="Zepto Support Assistant")

class AskRequest(BaseModel):
    query: str

@app.on_event("startup")
def startup():
    ensure_index()

@app.post("/ask", response_model=Answer)
def ask_endpoint(req: AskRequest):
    return ask(req.query)
