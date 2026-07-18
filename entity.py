from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body used by the chat and streaming endpoints."""

    message: str = Field(
        min_length=1,
        description="The user's travel request.",
    )

    session_id: str = Field(
        min_length=1,
        description="Unique browser conversation identifier.",
    )


class ChatResponse(BaseModel):
    """Normal non-streaming chat response."""

    response: str

    hotels: Optional[list[dict[str, Any]]] = None
    flights: Optional[list[dict[str, Any]]] = None


class BookingHistoryResponse(BaseModel):
    """Bookings belonging to one browser session."""

    session_id: str
    bookings: list[dict[str, Any]]


class ChatHistoryResponse(BaseModel):
    """Stored conversation history for one browser session."""

    session_id: str
    messages: list[dict[str, Any]]