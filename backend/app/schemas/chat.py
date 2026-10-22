
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


class UploadResponse(BaseModel):
    status: str
    message: str
    details: dict | None = None


class KBStatusResponse(BaseModel):
    status: str
    total_vectors: int | None = None

