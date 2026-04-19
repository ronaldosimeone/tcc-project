
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import time

router = APIRouter(
    prefix="/sse",
    tags=["stream"]
)

@router.get("")
def sse_stream():
    def event_generator():
        while True:
            yield f"data: ping {time.time()}\n\n"
            time.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
