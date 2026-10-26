import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from phoenix.otel import using_session
from pydantic import BaseModel
from agent import build_agent

load_dotenv()

agent = build_agent()

app = FastAPI(
    title="LangGraph Agent API",
    description="A simple API for interacting with a LangGraph agent that can search the web",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # omit to start a new conversation


class ChatResponse(BaseModel):
    response: str
    session_id: str  # send this back to continue the conversation


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message to the agent and get a response."""
    session_id = request.session_id or str(uuid.uuid4())
    # thread_id makes LangGraph load this conversation's history; using_session groups the traces in Phoenix
    with using_session(session_id):
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": session_id}},
        )
    return ChatResponse(response=result["messages"][-1].content, session_id=session_id)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
