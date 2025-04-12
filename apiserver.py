from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from agents.state import InitialState
from graph import run_langgraph


app = FastAPI()


@app.post("/refine-question")
async def send_notification(request: InitialState):
    return StreamingResponse(run_langgraph(request), media_type="text/event-stream")
