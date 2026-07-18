from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from .entity import GraphState
from .llm import llm
from .tools import search_flights, search_hotel


PLANNER_SYSTEM_PROMPT = """
You are TripWeaver's Trip Planner Agent.

Create a practical travel plan by combining real MCP flight and hotel results,
deterministic date and budget calculations, and concise travel guidance.

Rules:
- Never invent flight or hotel prices, schedules, availability, or booking IDs.
- Use only supplied MCP results for flights and hotels.
- Do not change supplied totals, dates, nights, rankings, or budget status.
- Treat flight prices as per traveller unless stated otherwise.
- Treat the user's budget as the total group budget unless stated otherwise.
- Produce exactly the requested number of itinerary days.
- Do not add a separate extra departure day.
- Use supplied weekdays and place weekend-only attractions only on weekends.
- Do not tell the user to search another website.
- Do not claim anything is booked.
- Do not expose internal provider IDs.
"""


CITY_TO_AIRPORT = {
    "abu dhabi": "AUH", "ahmedabad": "AMD", "amsterdam": "AMS",
    "bangalore": "BLR", "bengaluru": "BLR", "bangkok": "BKK",
    "beijing": "PEK", "berlin": "BER", "chennai": "MAA",
    "colombo": "CMB", "delhi": "DEL", "new delhi": "DEL",
    "doha": "DOH", "dubai": "DXB", "frankfurt": "FRA",
    "hong kong": "HKG", "hyderabad": "HYD", "istanbul": "IST",
    "jakarta": "CGK", "kathmandu": "KTM", "kuala lumpur": "KUL",
    "london": "LHR", "los angeles": "LAX", "male": "MLE",
    "malé": "MLE", "manila": "MNL", "melbourne": "MEL",
    "mumbai": "BOM", "muscat": "MCT", "new york": "JFK",
    "paris": "CDG", "riyadh": "RUH", "seoul": "ICN",
    "shanghai": "PVG", "singapore": "SIN", "sydney": "SYD",
    "tokyo": "NRT", "toronto": "YYZ",
}


def _extract_items(result: Any, key: str) -> list[dict]:
    if isinstance(result, dict):
        items = result.get(key, [])
        return items if isinstance(items, list) else []
    return result if isinstance(result, list) else []


def _tool_error(result: Any) -> Optional[str]:
    if isinstance(result, dict) and result.get("ok") is False:
        return str(result.get("error") or result.get("message") or
                   "The travel service is currently unavailable.")
    return None


def _airport_code(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 3 and text.isalpha():
        return text.upper()
    return CITY_TO_AIRPORT.get(text.casefold(), text.upper())


def _number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = "".join(c for c in str(value).replace(",", "")
                      if c.isdigit() or c in ".-")
    if not cleaned or cleaned in {"-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _money(value: Optional[float]) -> str:
    if value is None:
        return "Unavailable"
    return f"{int(value):,}" if value.is_integer() else f"{value:,.2f}"


def _flight_number(flight: dict) -> str:
    return str(flight.get("flightNumber") or flight.get("flight_number") or
               flight.get("flightNo") or "N/A")


def _flight_endpoint(value: Any) -> str:
    if isinstance(value, dict):
        city = value.get("city")
        airport = value.get("airport") or value.get("code")
        if city and airport:
            return f"{city} ({airport})"
        return str(city or airport or "Unknown")
    return str(value or "Unknown")


def _flight_price(flight: dict) -> Optional[float]:
    return _number(flight.get("price") or flight.get("fare") or
                   flight.get("amount"))


def _hotel_price(hotel: dict) -> Optional[float]:
    return _number(hotel.get("pricePerNight") or
                   hotel.get("price_per_night") or hotel.get("price"))


def _hotel_rating(hotel: dict) -> float:
    return _number(hotel.get("starRating") or hotel.get("stars") or
                   hotel.get("rating")) or 0.0


def _sort_flights(flights: list[dict]) -> list[dict]:
    return sorted(flights, key=lambda f: (
        _flight_price(f) is None,
        _flight_price(f) or float("inf"),
        str(f.get("departureTime") or f.get("departure_time") or ""),
    ))


def _sort_hotels(hotels: list[dict]) -> list[dict]:
    return sorted(hotels, key=lambda h: (
        _hotel_price(h) is None,
        _hotel_price(h) or float("inf"),
        -_hotel_rating(h),
    ))


def _format_flights(flights: list[dict], travellers: int, limit: int = 3) -> str:
    if not flights:
        return "No matching flights were returned by the flight service."
    rows = []
    for rank, flight in enumerate(flights[:limit], 1):
        airline = flight.get("airline", "Unknown airline")
        price = _flight_price(flight)
        total = price * travellers if price is not None else None
        currency = str(flight.get("currency") or "USD")
        date = (flight.get("flightDate") or flight.get("date") or
                flight.get("departure_date") or "Unknown date")
        departure = flight.get("departureTime") or flight.get("departure_time") or "Unknown"
        arrival = flight.get("arrivalTime") or flight.get("arrival_time") or "Unknown"
        rows.append(
            f"{rank}. {airline} {_flight_number(flight)}\n"
            f"   Route: {_flight_endpoint(flight.get('origin'))} → "
            f"{_flight_endpoint(flight.get('destination'))}\n"
            f"   Date and time: {date}, {departure}–{arrival}\n"
            f"   Price per traveller: {currency} {_money(price)}\n"
            f"   Total for {travellers} traveller(s): {currency} {_money(total)}"
        )
    return "\n".join(rows)


def _hotel_name(hotel: dict) -> str:
    return str(hotel.get("name") or hotel.get("hotelName") or
               hotel.get("hotel_name") or "Unknown hotel")


def _hotel_city(hotel: dict) -> str:
    city = hotel.get("city")
    location = hotel.get("location")
    if city is None and isinstance(location, dict):
        city = location.get("city")
    if isinstance(city, dict):
        return str(city.get("name") or city.get("city") or "Unknown city")
    return str(city or "Unknown city")


def _format_hotels(hotels: list[dict], nights: int, limit: int = 3) -> str:
    if not hotels:
        return "No matching hotels were returned by the hotel service."
    rows = []
    for rank, hotel in enumerate(hotels[:limit], 1):
        nightly = _hotel_price(hotel)
        total = nightly * nights if nightly is not None else None
        currency = str(hotel.get("currency") or "USD")
        label = "Best budget option" if rank == 1 else "Alternative option"
        rows.append(
            f"{rank}. {_hotel_name(hotel)} — {label}\n"
            f"   City: {_hotel_city(hotel)}\n"
            f"   Rating: {_money(_hotel_rating(hotel))} stars\n"
            f"   Price per night: {currency} {_money(nightly)}\n"
            f"   Total for {nights} night(s): {currency} {_money(total)}"
        )
    return "\n".join(rows)


def _trip_dates(travel_date: str, trip_days: int) -> tuple[str, str, int, list[str]]:
    start = datetime.strptime(travel_date, "%Y-%m-%d")
    safe_days = max(trip_days, 1)
    nights = max(safe_days - 1, 0)
    departure = start + timedelta(days=safe_days - 1)
    days = [
        f"Day {i + 1}: {(start + timedelta(days=i)).strftime('%A, %B %d, %Y')}"
        for i in range(safe_days)
    ]
    return start.date().isoformat(), departure.date().isoformat(), nights, days


def _calculate_check_out(check_in: str, nights: int) -> str:
    start = datetime.strptime(check_in, "%Y-%m-%d").date()
    return (start + timedelta(days=max(nights, 0))).isoformat()


def _budget_summary(budget: float, currency: str, travellers: int,
                    flights: list[dict], hotels: list[dict], nights: int) -> str:
    flight_price = _flight_price(flights[0]) if flights else None
    hotel_price = _hotel_price(hotels[0]) if hotels else None
    flight_total = flight_price * travellers if flight_price is not None else None
    hotel_total = hotel_price * nights if hotel_price is not None else None
    known_total = sum(v for v in (flight_total, hotel_total) if v is not None)
    remaining = budget - known_total

    if flight_total is None or hotel_total is None:
        status = "Cannot be fully verified because one or more required prices are unavailable."
    elif remaining >= 0:
        status = "Within budget before food, transport, activities, insurance, and personal expenses."
    else:
        status = (f"Over budget by {currency} {_money(abs(remaining))} before food, "
                  "transport, activities, insurance, and personal expenses.")

    return (
        f"- Total group budget: {currency} {_money(budget)}\n"
        f"- Budget per traveller: {currency} {_money(budget / travellers)}\n"
        f"- Cheapest flight total for {travellers} traveller(s): {currency} {_money(flight_total)}\n"
        f"- Cheapest hotel total for {nights} night(s): {currency} {_money(hotel_total)}\n"
        f"- Known flight and hotel total: {currency} {_money(known_total)}\n"
        f"- Remaining after known costs: {currency} {_money(remaining)}\n"
        f"- Budget status: {status}"
    )


def _missing_details(state: GraphState) -> list[str]:
    required = [
        ("departure city or airport", state.get("origin")),
        ("destination city", state.get("destination")),
        ("travel date", state.get("flight_date")),
        ("trip duration in days", state.get("trip_days")),
        ("number of travellers", state.get("travellers")),
        ("approximate total group budget", state.get("budget")),
    ]
    return [label for label, value in required if value in (None, "", 0)]


def planner_node(state: GraphState) -> dict:
    missing = _missing_details(state)
    if missing:
        return {
            "hotel_results": [],
            "flight_results": [],
            "response_text": (
                "## Let’s plan your trip\n\nPlease provide:\n\n"
                + "\n".join(f"- {item}" for item in missing)
                + "\n\nExample:\n\n`From Colombo to Bangkok on 2026-08-20, "
                  "5 days, 2 travellers, budget USD 1,200`"
            ),
        }

    origin_name = str(state.get("origin")).strip()
    destination_name = str(state.get("destination")).strip()
    origin_code = _airport_code(origin_name)
    destination_code = _airport_code(destination_name)
    travel_date = str(state.get("flight_date")).strip()
    trip_days = max(int(state.get("trip_days") or 1), 1)
    travellers = max(int(state.get("travellers") or 1), 1)
    budget = float(state.get("budget") or 0)
    currency = str(state.get("budget_currency") or "USD").upper()

    check_in, departure_date, nights, itinerary_dates = _trip_dates(travel_date, trip_days)
    check_in = str(state.get("check_in") or check_in)
    check_out = str(state.get("check_out") or _calculate_check_out(check_in, nights))

    flight_result = search_flights.invoke({
        "origin": origin_code,
        "destination": destination_code,
        "date": travel_date,
    })

    hotel_params = {"city": destination_name, "checkIn": check_in, "checkOut": check_out}
    hotel_result = search_hotel.invoke(hotel_params)

    flight_error = _tool_error(flight_result)
    hotel_error = _tool_error(hotel_result)
    flights = [] if flight_error else _extract_items(flight_result, "flights")
    hotels = [] if hotel_error else _extract_items(hotel_result, "hotels")
    flights = _sort_flights(flights)
    hotels = _sort_hotels(hotels)

    notes = []
    if flight_error:
        notes.append(f"Flight service: {flight_error}")
    if hotel_error:
        notes.append(f"Hotel service: {hotel_error}")
    if not flights and not flight_error:
        notes.append(f"No flight matched {origin_code} → {destination_code} on {travel_date}.")
    if not hotels and not hotel_error:
        notes.append(f"No hotel matched {destination_name} for {check_in} to {check_out}.")
    service_notes = "\n".join(f"- {n}" for n in notes) if notes else "No service errors."

    budget_text = _budget_summary(
        budget, currency, travellers, flights, hotels, nights
    )
    itinerary_text = "\n".join(f"- {line}" for line in itinerary_dates)

    prompt = f"""
Create a concise trip plan using these fixed facts.

TRIP FACTS:
- Origin: {origin_name}
- Destination: {destination_name}
- Flight search route: {origin_code} → {destination_code}
- Start date: {travel_date}
- Final itinerary/departure date: {departure_date}
- Duration: exactly {trip_days} days
- Hotel stay: exactly {nights} nights
- Travellers: {travellers}
- Total group budget: {currency} {_money(budget)}
- Budget per traveller: {currency} {_money(budget / travellers)}
- Hotel check-in: {check_in}
- Hotel check-out: {check_out}

EXACT ITINERARY DAYS:
{itinerary_text}

RANKED REAL FLIGHT RESULTS:
{_format_flights(flights, travellers)}

RANKED REAL HOTEL RESULTS:
{_format_hotels(hotels, nights)}

DETERMINISTIC BUDGET CALCULATION:
{budget_text}

SERVICE NOTES:
{service_notes}

Return these sections:
1. Trip overview
2. Recommended flight options
3. Recommended hotel options
4. Budget status
5. Day-by-day itinerary
6. Practical travel tips
7. Next steps inside TripWeaver

Important:
- Produce exactly {trip_days} itinerary days.
- Day {trip_days} is the final/departure day.
- Do not create Day {trip_days + 1}.
- Preserve all rankings and calculations.
- Show flight price per traveller and total for all travellers.
- Show hotel price per night and total for all nights.
- Treat the budget as the total group budget.
- Do not invent any service data.
- If no flight exists, say no MCP result matched the exact route and date.
- Suggest another date or route inside TripWeaver, not another website.
"""

    try:
        response = llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        response_text = str(getattr(response, "content", "")).strip()
        if not response_text:
            raise ValueError("The planner returned an empty response.")
    except Exception as exc:
        print(f"Planner Agent error: {type(exc).__name__}: {exc}")
        response_text = (
            "## Trip plan unavailable\n\n"
            f"### Flights\n{_format_flights(flights, travellers)}\n\n"
            f"### Hotels\n{_format_hotels(hotels, nights)}\n\n"
            f"### Budget\n{budget_text}"
        )

    return {
        "hotel_results": hotels,
        "flight_results": flights,
        "response_text": response_text,
    }