from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import tool

from .mcp_client import call_mcp_tool


@tool
def get_hotels() -> List[dict] | dict:
    """Get all hotels through the Hotel MCP server."""
    result = call_mcp_tool("hotel", "list_hotels", {})
    if isinstance(result, dict) and result.get("ok") is False:
        return result
    return result.get("hotels", []) if isinstance(result, dict) else []


@tool
def search_hotel(
    city: str,
    checkIn: Optional[str] = None,
    checkOut: Optional[str] = None,
) -> List[dict] | dict:
    """Search hotels through the Hotel MCP server."""
    result = call_mcp_tool(
        "hotel",
        "search_hotels",
        {"city": city, "check_in": checkIn, "check_out": checkOut},
    )
    if isinstance(result, dict) and result.get("ok") is False:
        return result
    return result.get("hotels", []) if isinstance(result, dict) else []


@tool
def book_hotel(
    hotel_id: str,
    guest_name: str,
    guest_email: str,
    check_in_date: str,
    check_out_date: str,
    room_type: str,
) -> dict:
    """Book a hotel through the Hotel MCP server."""
    return call_mcp_tool(
        "hotel",
        "book_hotel",
        {
            "hotel_id": hotel_id,
            "guest_name": guest_name,
            "guest_email": guest_email,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "room_type": room_type,
        },
    )


@tool
def get_flights() -> List[dict] | dict:
    """Get all flights through the Flight MCP server."""
    result = call_mcp_tool("flight", "list_flights", {})
    if isinstance(result, dict) and result.get("ok") is False:
        return result
    return result.get("flights", []) if isinstance(result, dict) else []


@tool
def search_flights(
    origin: str,
    destination: str,
    date: Optional[str] = None,
) -> List[dict] | dict:
    """Search flights through the Flight MCP server."""
    result = call_mcp_tool(
        "flight",
        "search_flights",
        {"origin": origin, "destination": destination, "date": date},
    )
    if isinstance(result, dict) and result.get("ok") is False:
        return result
    return result.get("flights", []) if isinstance(result, dict) else []


@tool
def book_flight(flight_id: str, passenger_name: str, passenger_email: str) -> dict:
    """Book a flight through the Flight MCP server."""
    return call_mcp_tool(
        "flight",
        "book_flight",
        {
            "flight_id": flight_id,
            "passenger_name": passenger_name,
            "passenger_email": passenger_email,
        },
    )
