# app.py  (FastAPI)
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from qa_engine import ProductQASystem

CSV_PATH = os.environ.get("SHOPIFY_CSV_PATH", "products_export.csv")
qa = ProductQASystem(csv_path=CSV_PATH)

app = FastAPI(title="Rapid Medical QA", version="0.2.0")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    ok: bool
    answer: str
    product: Optional[str] = None
    attribute: Optional[str] = None
    value: Optional[str] = None
    suggestions: Optional[List[str]] = None
    confidence: Optional[int] = None

@app.get("/health")
def health():
    return {"ok": True, "mode": os.environ.get("SHOPIFY_MODE","csv")}

@app.post("/reload")
def reload_data():
    qa.reload()
    return {"ok": True, "message": "Data reloaded"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    result = qa.answer(req.question)
    return AskResponse(**result)
