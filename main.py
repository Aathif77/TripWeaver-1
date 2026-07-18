import json
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agents.graph import graph
from agents.tools import get_flights, get_hotels
from database import (
    clear_chat_history,
    get_booking_by_reference,
    get_bookings,
    get_chat_history,
    get_recent_chat_messages,
    initialize_database,
    save_booking,
    save_chat_message,
)
from entity import (
    BookingHistoryResponse,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    print("Database initialized")
    yield


app = FastAPI(
    title="TripWeaver API",
    description="MCP-Based Multi-Agent Travel Planner",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_initial_state(
    message: str,
    session_id: str,
) -> dict[str, Any]:
    """Build the shared LangGraph state using recent session history."""

    previous_messages = get_recent_chat_messages(
        session_id=session_id,
        limit=8,
    )

    flattened_messages = [
        item["content"]
        for item in previous_messages
    ]
    flattened_messages.append(message)

    return {
        "messages": flattened_messages,
        "session_id": session_id,
        "intent": "",
        "sub_action": "",
        "city": None,
        "check_in": None,
        "check_out": None,
        "hotel_id": None,
        "hotel_name": None,
        "guest_name": None,
        "guest_email": None,
        "room_type": None,
        "origin": None,
        "destination": None,
        "flight_date": None,
        "trip_days": None,
        "travellers": None,
        "budget": None,
        "budget_currency": "USD",
        "flight_id": None,
        "flight_number": None,
        "airline": None,
        "passenger_name": None,
        "passenger_email": None,
        "hotel_results": [],
        "flight_results": [],
        "response_text": "",
        "booking_record": None,
    }


def save_workflow_result(
    session_id: str,
    message: str,
    result: dict[str, Any],
) -> str:
    """Persist the conversation and return the final visible response."""

    response_text = result.get(
        "response_text",
        "Something went wrong. Please try again.",
    )

    save_chat_message(
        session_id=session_id,
        role="user",
        content=message,
    )

    booking_record = result.get("booking_record")

    if booking_record:
        booking_reference = save_booking(
            session_id=session_id,
            booking_type=booking_record.get(
                "booking_type",
                "unknown",
            ),
            status=booking_record.get(
                "status",
                "confirmed",
            ),
            provider_id=booking_record.get("provider_id"),
            public_reference=booking_record.get(
                "public_reference"
            ),
            customer_name=booking_record.get(
                "customer_name"
            ),
            customer_email=booking_record.get(
                "customer_email"
            ),
            confirmation_reference=booking_record.get(
                "confirmation_reference"
            ),
            booking_details=booking_record.get(
                "details",
                {},
            ),
        )

        response_text += (
            "\n\n---\n\n"
            "### TripWeaver Booking Reference\n\n"
            f"`{booking_reference}`\n\n"
            "Keep this reference to retrieve your booking later."
        )

    save_chat_message(
        session_id=session_id,
        role="assistant",
        content=response_text,
    )

    return response_text


_BOOKING_REFERENCE_PATTERN = re.compile(
    r"\bTW-\d{4}-[FHB]-[A-Z0-9]{6}\b",
    re.IGNORECASE,
)


def extract_booking_reference(message: str) -> str | None:
    match = _BOOKING_REFERENCE_PATTERN.search(message)
    return match.group(0).upper() if match else None


def format_booking_lookup(booking: dict) -> str:
    details = booking.get("details") or {}
    booking_type = str(
        booking.get("booking_type", "")
    ).lower()

    common = (
        f"## Booking {booking['booking_reference']}\n\n"
        f"- **Status:** {booking.get('status', 'Unknown')}\n"
        f"- **Customer:** "
        f"{booking.get('customer_name', 'Not available')}\n"
        f"- **Email:** "
        f"{booking.get('customer_email', 'Not available')}\n"
        f"- **Created:** {booking.get('created_at', 'Not available')}\n"
    )

    if booking_type == "flight":
        return (
            common
            + f"- **Type:** Flight\n"
            + f"- **Airline:** "
            f"{details.get('airline', 'Unknown')}\n"
            + f"- **Flight:** "
            f"{details.get('flight_number', 'Unknown')}\n"
            + f"- **Route:** "
            f"{details.get('origin', 'Unknown')} → "
            f"{details.get('destination', 'Unknown')}\n"
            + f"- **Date:** "
            f"{details.get('flight_date', 'Unknown')}\n"
            + f"- **Price:** "
            f"{details.get('currency', 'USD')} "
            f"{details.get('price', 'N/A')}\n"
        )

    if booking_type == "hotel":
        return (
            common
            + f"- **Type:** Hotel\n"
            + f"- **Hotel:** "
            f"{details.get('hotel_name', 'Unknown')}\n"
            + f"- **City:** "
            f"{details.get('city', 'Unknown')}\n"
            + f"- **Check-in:** "
            f"{details.get('check_in', 'Unknown')}\n"
            + f"- **Check-out:** "
            f"{details.get('check_out', 'Unknown')}\n"
        )

    return common + f"- **Type:** {booking_type or 'Unknown'}\n"


def handle_booking_lookup(
    message: str,
    session_id: str,
) -> str | None:
    booking_reference = extract_booking_reference(message)

    if not booking_reference:
        return None

    booking = get_booking_by_reference(
        session_id=session_id,
        booking_reference=booking_reference,
    )

    if booking is None:
        response_text = (
            "## Booking not found\n\n"
            f"I could not find `{booking_reference}` in this session.\n\n"
            "Check the reference or open the browser session that "
            "created the booking."
        )
    else:
        response_text = format_booking_lookup(booking)

    save_chat_message(
        session_id=session_id,
        role="user",
        content=message,
    )
    save_chat_message(
        session_id=session_id,
        role="assistant",
        content=response_text,
    )

    return response_text

def validate_request(request: ChatRequest) -> tuple[str, str]:
    message = request.message.strip()
    session_id = request.session_id.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    if not session_id:
        raise HTTPException(
            status_code=400,
            detail="Session ID cannot be empty.",
        )

    return message, session_id


def stream_event(
    event_type: str,
    **payload: Any,
) -> bytes:
    """Encode one newline-delimited JSON streaming event."""

    event = {
        "type": event_type,
        **payload,
    }
    return (
        json.dumps(event, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def response_tokens(text: str) -> Iterator[str]:
    """Split Markdown into small chunks while preserving whitespace."""

    for token in re.findall(r"\S+\s*|\n", text):
        yield token


def stream_workflow(
    message: str,
    session_id: str,
) -> Iterator[bytes]:
    """
    Stream activity events immediately, run the LangGraph workflow,
    then stream the final response progressively.
    """

    try:
        yield stream_event(
            "status",
            stage="routing",
            message="Understanding your request...",
        )

        lookup_response = handle_booking_lookup(
            message=message,
            session_id=session_id,
        )

        if lookup_response is not None:
            yield stream_event(
                "status",
                stage="lookup",
                message="Retrieving your booking...",
            )

            for token in response_tokens(lookup_response):
                yield stream_event(
                    "token",
                    content=token,
                )
                time.sleep(0.018)

            yield stream_event(
                "done",
                response=lookup_response,
            )
            return

        initial_state = build_initial_state(
            message=message,
            session_id=session_id,
        )

        yield stream_event(
            "status",
            stage="working",
            message="Checking the right travel service...",
        )

        result = graph.invoke(initial_state)

        intent = result.get("intent", "general")
        sub_action = result.get("sub_action", "general")

        if sub_action == "book":
            activity_message = "Confirming your booking..."
        elif intent == "hotel":
            activity_message = "Preparing hotel options..."
        elif intent == "flight":
            activity_message = "Preparing flight options..."
        elif intent == "planner":
            activity_message = (
                "Combining flights, hotels, budget, and itinerary..."
            )
        else:
            activity_message = "Preparing travel guidance..."

        yield stream_event(
            "status",
            stage="responding",
            message=activity_message,
        )

        response_text = save_workflow_result(
            session_id=session_id,
            message=message,
            result=result,
        )

        for token in response_tokens(response_text):
            yield stream_event(
                "token",
                content=token,
            )
            time.sleep(0.018)

        yield stream_event(
            "done",
            response=response_text,
        )

    except Exception as exc:
        print(
            "Streaming workflow failed: "
            f"{type(exc).__name__}: {exc}"
        )

        yield stream_event(
            "error",
            message=(
                "TripWeaver could not complete that request. "
                "Please try again."
            ),
        )


@app.get("/")
async def root():
    return {
        "application": "TripWeaver",
        "status": "online",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/hotels")
async def list_hotels():
    return get_hotels.invoke({})


@app.get("/flights")
async def list_flights():
    return get_flights.invoke({})


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Original non-streaming endpoint retained for compatibility."""

    message, session_id = validate_request(request)

    lookup_response = handle_booking_lookup(
        message=message,
        session_id=session_id,
    )

    if lookup_response is not None:
        return ChatResponse(
            response=lookup_response,
            hotels=None,
            flights=None,
        )

    try:
        result = graph.invoke(
            build_initial_state(
                message=message,
                session_id=session_id,
            )
        )
    except Exception as exc:
        print(
            "Graph execution failed: "
            f"{type(exc).__name__}: {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail="The travel workflow could not be completed.",
        ) from exc

    response_text = save_workflow_result(
        session_id=session_id,
        message=message,
        result=result,
    )

    return ChatResponse(
        response=response_text,
        hotels=result.get("hotel_results") or None,
        flights=result.get("flight_results") or None,
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream progress events and the final reply as NDJSON."""

    message, session_id = validate_request(request)

    return StreamingResponse(
        stream_workflow(
            message=message,
            session_id=session_id,
        ),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/chat-history/{session_id}",
    response_model=ChatHistoryResponse,
)
async def read_chat_history(session_id: str):
    return ChatHistoryResponse(
        session_id=session_id,
        messages=get_chat_history(session_id),
    )


@app.delete("/chat-history/{session_id}")
async def delete_chat_history(session_id: str):
    clear_chat_history(session_id)

    return {
        "message": "Conversation cleared.",
        "session_id": session_id,
    }


@app.get(
    "/bookings/{session_id}",
    response_model=BookingHistoryResponse,
)
async def read_bookings(session_id: str):
    return BookingHistoryResponse(
        session_id=session_id,
        bookings=get_bookings(session_id),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )