from __future__ import annotations

from typing import Any, Optional
import json
import requests
from mcp.server.fastmcp import FastMCP

FLIGHT_API_BASE = "https://standing-fish-574.convex.site/flights"
REQUEST_TIMEOUT = 30  # Increased timeout

mcp = FastMCP("TripWeaver Flight Service")


def _request_json(method: str, url: str, **kwargs: Any) -> Any:
    """Make a JSON request with better error handling."""
    try:
        print(f"Making {method} request to: {url}")
        if 'json' in kwargs:
            print(f"Payload: {json.dumps(kwargs['json'], indent=2)}")
        
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text[:500]}")  # Log first 500 chars
        
        response.raise_for_status()
        
        if not response.text:
            return {"ok": False, "error": "Empty response from server"}
            
        return response.json()
        
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Flight service timed out. Please try again."}
    except requests.exceptions.HTTPError as exc:
        error_msg = f"Flight service error: {exc}"
        try:
            # Try to get more details from the response
            if hasattr(exc, 'response') and exc.response:
                error_data = exc.response.json()
                if isinstance(error_data, dict) and error_data.get('error'):
                    error_msg = f"Flight service error: {error_data.get('error')}"
                elif isinstance(error_data, dict) and error_data.get('message'):
                    error_msg = f"Flight service error: {error_data.get('message')}"
        except:
            pass
        return {"ok": False, "error": error_msg}
    except requests.exceptions.RequestException as exc:
        return {"ok": False, "error": f"Flight service unavailable: {exc}"}
    except ValueError as exc:
        return {"ok": False, "error": f"Invalid JSON response: {exc}"}


def _normalise_place(value: str) -> str:
    return value.upper() if len(value) == 3 and value.isalpha() else value


@mcp.tool()
def list_flights() -> dict:
    """List every currently available flight."""
    data = _request_json("GET", FLIGHT_API_BASE)
    if isinstance(data, dict) and data.get("ok") is False:
        return data
    
    # Handle different response formats
    if isinstance(data, dict):
        flights = data.get("flights", [])
        if not flights and "data" in data:
            flights = data.get("data", [])
    else:
        flights = []
    
    return {"ok": True, "flights": flights}


@mcp.tool()
def search_flights(
    origin: str,
    destination: str,
    date: Optional[str] = None,
) -> dict:
    """Search flights by origin, destination, and optional travel date."""
    origin = _normalise_place(origin)
    destination = _normalise_place(destination)
    
    params = {
        "origin": origin,
        "destination": destination,
    }
    if date:
        params["date"] = date

    data = _request_json("GET", f"{FLIGHT_API_BASE}/search", params=params)
    if isinstance(data, dict) and data.get("ok") is False:
        return data
    
    # Handle different response formats
    if isinstance(data, dict):
        flights = data.get("flights", [])
        if not flights and "data" in data:
            flights = data.get("data", [])
    else:
        flights = []
    
    return {"ok": True, "flights": flights}


@mcp.tool()
def book_flight(
    flight_id: str,
    passenger_name: str,
    passenger_email: str,
) -> dict:
    """Book a flight and return the provider confirmation."""
    # Validate inputs
    if not flight_id:
        return {"ok": False, "error": "Flight ID is required"}
    
    if not passenger_name or not passenger_name.strip():
        return {"ok": False, "error": "Passenger name is required"}
    
    if not passenger_email or not passenger_email.strip():
        return {"ok": False, "error": "Passenger email is required"}
    
    # Validate email format (basic)
    if "@" not in passenger_email or "." not in passenger_email:
        return {"ok": False, "error": "Invalid email format"}
    
    # Prepare payload - try both formats
    payload = {
        "flightId": flight_id,  # Original format
        "passengerName": passenger_name.strip(),
        "passengerEmail": passenger_email.strip(),
    }
    
    # Try with alternative field names if the first fails
    try:
        data = _request_json("POST", f"{FLIGHT_API_BASE}/book", json=payload)
        if isinstance(data, dict):
            # Check if we got a successful response
            if data.get("ok") is True:
                return data
            elif data.get("error"):
                # If error, try alternative format
                alt_payload = {
                    "id": flight_id,
                    "name": passenger_name.strip(),
                    "email": passenger_email.strip(),
                }
                alt_data = _request_json("POST", f"{FLIGHT_API_BASE}/book", json=alt_payload)
                if isinstance(alt_data, dict):
                    alt_data.setdefault("ok", alt_data.get("error") is None)
                    return alt_data
                return data
            else:
                data.setdefault("ok", data.get("error") is None)
                return data
        return {"ok": False, "error": "Flight booking service returned an unexpected response."}
    
    except Exception as e:
        return {"ok": False, "error": f"Booking failed: {str(e)}"}


@mcp.tool()
def get_flight_details(flight_id: str) -> dict:
    """Get details for a specific flight."""
    if not flight_id:
        return {"ok": False, "error": "Flight ID is required"}
    
    data = _request_json("GET", f"{FLIGHT_API_BASE}/{flight_id}")
    if isinstance(data, dict) and data.get("ok") is False:
        return data
    
    # Handle different response formats
    if isinstance(data, dict):
        if "flight" in data:
            return {"ok": True, "flight": data["flight"]}
        elif "data" in data:
            return {"ok": True, "flight": data["data"]}
        else:
            return {"ok": True, "flight": data}
    
    return {"ok": False, "error": "Invalid response format"}


@mcp.tool()
def get_flight_by_number(flight_number: str) -> dict:
    """Get flight details by flight number."""
    if not flight_number:
        return {"ok": False, "error": "Flight number is required"}
    
    data = _request_json("GET", f"{FLIGHT_API_BASE}/number/{flight_number}")
    if isinstance(data, dict) and data.get("ok") is False:
        return data
    
    # Handle different response formats
    if isinstance(data, dict):
        if "flight" in data:
            return {"ok": True, "flight": data["flight"]}
        elif "data" in data:
            return {"ok": True, "flight": data["data"]}
        else:
            return {"ok": True, "flight": data}
    
    return {"ok": False, "error": "Invalid response format"}


if __name__ == "__main__":
    mcp.run(transport="stdio")