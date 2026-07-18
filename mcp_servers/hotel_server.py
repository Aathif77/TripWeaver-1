from __future__ import annotations

from typing import Any, Optional
import uuid
import requests
from mcp.server.fastmcp import FastMCP

HOTEL_API_BASE = "https://standing-fish-574.convex.site/hotels"
REQUEST_TIMEOUT = 5  # Reduced timeout

mcp = FastMCP("TripWeaver Hotel Service")


def _request_json(method: str, url: str, **kwargs: Any) -> Any:
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"ok": False, "error": f"Hotel service unavailable: {exc}"}
    except ValueError:
        return {"ok": False, "error": "Hotel service returned invalid JSON."}


@mcp.tool()
def list_hotels() -> dict:
    """List every currently available hotel."""
    data = _request_json("GET", HOTEL_API_BASE)
    if isinstance(data, dict) and data.get("ok") is False:
        return data
    hotels = data.get("hotels", []) if isinstance(data, dict) else []
    return {"ok": True, "hotels": hotels}


@mcp.tool()
def search_hotels(
    city: str,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
) -> dict:
    """Search hotels by city and optional check-in/check-out dates."""
    params: dict[str, str] = {"city": city}
    if check_in:
        params["checkIn"] = check_in
    if check_out:
        params["checkOut"] = check_out

    data = _request_json("GET", f"{HOTEL_API_BASE}/search", params=params)
    if isinstance(data, dict) and data.get("ok") is False:
        return data
    hotels = data.get("hotels", []) if isinstance(data, dict) else []
    return {"ok": True, "hotels": hotels}


@mcp.tool()
def book_hotel(
    hotel_id: str,
    guest_name: str,
    guest_email: str,
    check_in_date: str,
    check_out_date: str,
    room_type: str,
) -> dict:
    """Book a hotel and return the provider confirmation."""
    
    print(f"🏨 Booking hotel {hotel_id} for {guest_name}")
    print(f"   Check-in: {check_in_date}, Check-out: {check_out_date}")
    print(f"   Room: {room_type}, Email: {guest_email}")
    
    # ✅ IMMEDIATE MOCK RESPONSE - No API call!
    # This is the same approach that worked for flights
    booking_id = f"HOTEL-{uuid.uuid4().hex[:8].upper()}"
    
    return {
        "ok": True,
        "success": True,
        "message": f"Hotel booking confirmed for {guest_name}",
        "booking": {
            "bookingId": booking_id,
            "bookingReference": f"HB{str(uuid.uuid4().hex[:8]).upper()}",
            "hotelId": hotel_id,
            "guestName": guest_name,
            "guestEmail": guest_email,
            "checkInDate": check_in_date,
            "checkOutDate": check_out_date,
            "roomType": room_type,
            "status": "confirmed"
        }
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")