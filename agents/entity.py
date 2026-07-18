from typing import Any, List, Optional, TypedDict


class GraphState(TypedDict, total=False):
    """Shared state passed between TripWeaver LangGraph nodes."""

    session_id: str
    messages: List[str]

    intent: str
    sub_action: str

    # Hotel search and booking
    city: Optional[str]
    check_in: Optional[str]
    check_out: Optional[str]

    hotel_id: Optional[str]
    hotel_name: Optional[str]

    guest_name: Optional[str]
    guest_email: Optional[str]

    # Kept temporarily because the current hotel tool still expects it.
    room_type: Optional[str]

    # Flight search and booking
    origin: Optional[str]
    destination: Optional[str]
    flight_date: Optional[str]

    flight_id: Optional[str]
    flight_number: Optional[str]
    airline: Optional[str]

    passenger_name: Optional[str]
    passenger_email: Optional[str]

    # Complete trip planning
    trip_days: Optional[int]
    travellers: Optional[int]

    budget: Optional[float]
    budget_currency: Optional[str]

    # Workflow results
    hotel_results: List[dict]
    flight_results: List[dict]

    response_text: str

    # Returned after a successful hotel or flight booking.
    #
    # main.py receives this record, saves it into SQLite and then
    # adds the generated TripWeaver reference to the response.
    booking_record: Optional[dict[str, Any]]