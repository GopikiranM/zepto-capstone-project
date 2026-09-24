from __future__ import annotations
import os
import json
from pathlib import Path
from typing import TypedDict
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
import chromadb
from langgraph.graph import StateGraph, END

ROOT=Path("/content/zepto_capstone_project/support_assistant")
DOCS=ROOT/"docs"; DB=ROOT/"chroma_db"; COLLECTION="zepto_policy"
MODEL_NAME="all-MiniLM-L6-v2"
KEYWORDS=["delivery","return","refund","membership","tracking","cancel","gift card","support hours"]
PROMPT_TEMPLATE="""ROLE: You are a Zepto policy support assistant.\nCONTEXT: Use only the retrieved Zepto policy context supplied below.\nTASK: Answer the user's question using that context.\nFORMAT: Return a JSON object with answer, sources, and confidence.\nLENGTH: Keep the answer concise and directly relevant.\nNEGATIVE CONSTRAINT: Do not answer using information not present in the provided context.\nFEW-SHOT EXAMPLE: User: What is the delivery fee below INR 149? Context: Orders below INR 149 incur INR 25. Assistant: {{\"answer\":\"INR 25.\",\"sources\":[\"doc_01\"],\"confidence\":1.0}}\nCONTEXT:\n{context}\nUSER QUERY:\n{query}"""

class Answer(BaseModel):
    answer:str
    sources:list[str]=Field(default_factory=list)
    confidence:float=Field(ge=0,le=1)
class State(TypedDict, total=False):
    query:str; intent:str; retrieved:list[dict]; answer:Answer

def mock_llm(): return os.getenv("MOCK_LLM","1")!="0"

def get_collection():
    client=chromadb.PersistentClient(path=str(DB)); return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space":"cosine"})

def ensure_index():
    col=get_collection()
    if col.count() >= 8: return col
    model=SentenceTransformer(MODEL_NAME)
    ids=[]; docs=[]; metas=[]; embs=[]
    for p in sorted(DOCS.glob("doc_*.txt")):
        text=p.read_text(encoding="utf-8").strip(); ids.append(p.stem); docs.append(text); metas.append({"document_id":p.stem}); embs.append(model.encode(text,normalize_embeddings=True).tolist())
    col.upsert(ids=ids,documents=docs,metadatas=metas,embeddings=embs); return col

def classify_intent(state:State)->State:
    q=state["query"].lower(); intent="policy_question" if any(k in q for k in KEYWORDS) else "general_question"; return {**state,"intent":intent}

def retrieve_and_answer(state:State)->State:
    model=SentenceTransformer(MODEL_NAME); col=ensure_index(); emb=model.encode(state["query"],normalize_embeddings=True).tolist(); res=col.query(query_embeddings=[emb],n_results=3)
    retrieved=[]
    for i,doc in enumerate(res["documents"][0]): retrieved.append({"id":res["ids"][0][i],"document":doc})
    top=retrieved[0]; snippet=top["document"][:200]
    if mock_llm(): ans=f"Based on the retrieved context: {snippet}"; out=Answer(answer=ans,sources=[x["id"] for x in retrieved],confidence=1.0)
    else: out=real_llm_answer(state["query"],retrieved)
    return {**state,"retrieved":retrieved,"answer":out}

def direct_answer(state:State)->State:
    if mock_llm(): out=Answer(answer="I can only answer questions about Zepto policies right now.",sources=[],confidence=1.0)
    else: out=real_llm_answer(state["query"],[])
    return {**state,"answer":out}

def real_llm_answer(query, retrieved):
    context = "\n\n".join(x["document"] for x in retrieved)
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    last_error = None

    # Optional real-LLM extension point.
    # Validate the raw LLM output and retry up to 2 additional times
    # when the output does not match the required Pydantic schema.
    for attempt in range(3):
        try:
            raw_output = os.getenv("LLM_RAW_OUTPUT", "")

            if not raw_output:
                raise RuntimeError(
                    "MOCK_LLM=0 is an optional extension; configure your LLM provider implementation here."
                )

            data = json.loads(raw_output)
            return Answer.model_validate(data)

        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc

            if attempt < 2:
                prompt = (
                    PROMPT_TEMPLATE.format(context=context, query=query)
                    + "\nReturn ONLY valid JSON matching the required answer, sources, and confidence fields."
                )

    raise RuntimeError(
        f"LLM output failed validation after 3 attempts: {last_error}"
    )

def build_graph():
    g=StateGraph(State); g.add_node("classify_intent",classify_intent); g.add_node("retrieve_and_answer",retrieve_and_answer); g.add_node("direct_answer",direct_answer); g.set_entry_point("classify_intent"); g.add_conditional_edges("classify_intent",lambda s:s["intent"],{"policy_question":"retrieve_and_answer","general_question":"direct_answer"}); g.add_edge("retrieve_and_answer",END); g.add_edge("direct_answer",END); return g.compile()
GRAPH=build_graph()

def ask(query:str)->Answer: return GRAPH.invoke({"query":query})["answer"]
