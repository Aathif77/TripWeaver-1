from typing import Literal, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from .entity import GraphState
from .llm import llm
from .prompts import get_system_prompt_for_general_qa, get_system_prompt_with_history
from .tools import book_flight, book_hotel, get_flights, get_hotels, search_flights, search_hotel

# ============================================================
# Structured Intent Extraction
# ============================================================

class TravelExtraction(BaseModel):
    intent: Literal["hotel", "flight", "general", "planner"] = Field(
        default="general"
    )
    sub_action: Literal["search", "list_all", "book", "general"] = Field(default="general")
    city: Optional[str] = Field(default=None, description="City name for hotel search")
    check_in: Optional[str] = Field(default=None, description="Check-in date YYYY-MM-DD")
    check_out: Optional[str] = Field(default=None, description="Check-out date YYYY-MM-DD")
    hotel_id: Optional[str] = Field(default=None, description="Internal hotel ID")
    hotel_name: Optional[str] = Field(default=None, description="Hotel name mentioned")
    guest_name: Optional[str] = Field(default=None, description="Guest full name")
    guest_email: Optional[str] = Field(default=None, description="Guest email")
    origin: Optional[str] = Field(default=None, description="Flight origin city or code")
    destination: Optional[str] = Field(default=None, description="Flight destination city or code")
    flight_date: Optional[str] = Field(default=None, description="Flight date YYYY-MM-DD")
    flight_id: Optional[str] = Field(default=None, description="Internal flight ID")
    flight_number: Optional[str] = Field(default=None, description="Flight number like CE2605")
    airline: Optional[str] = Field(default=None, description="Airline name mentioned")
    passenger_name: Optional[str] = Field(default=None, description="Passenger full name")
    passenger_email: Optional[str] = Field(
        default=None,
        description="Passenger email",
    )

    # Complete trip-planning fields
    trip_days: Optional[int] = Field(
        default=None,
        description="Number of days in the requested trip.",
    )
    travellers: Optional[int] = Field(
        default=None,
        description="Number of travellers.",
    )
    budget: Optional[float] = Field(
        default=None,
        description="Approximate numeric trip budget.",
    )
    budget_currency: Optional[str] = Field(
        default="USD",
        description="Budget currency such as USD, LKR, EUR, or GBP.",
    )

# We'll import llm inside functions to avoid circular imports
def get_llm():
    from .llm import llm
    return llm

travel_extractor = None

def get_travel_extractor():
    global travel_extractor
    if travel_extractor is None:
        from .llm import llm
        travel_extractor = llm.with_structured_output(TravelExtraction)
    return travel_extractor

# ============================================================
# Router
# ============================================================

def router(state: GraphState) -> dict:
    messages = state["messages"]
    user_msg = messages[-1]
    history = messages[:-1]
    
    session_id = state.get("session_id")
    print(f"🔍 Router - session_id: {session_id}")
    
    from .prompts import get_system_prompt_with_history
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    
    sys_prompt = get_system_prompt_with_history("\n".join(history))
    inv_msgs = [SystemMessage(content=sys_prompt)]
    
    for i in range(0, len(history), 2):
        inv_msgs.append(HumanMessage(content=history[i]))
        if i + 1 < len(history):
            inv_msgs.append(AIMessage(content=history[i + 1]))
    
    inv_msgs.append(HumanMessage(content=user_msg))
    
    try:
        extractor = get_travel_extractor()
        extracted = extractor.invoke(inv_msgs)
        data = extracted.model_dump() if hasattr(extracted, "model_dump") else extracted.dict()
    except Exception as exc:
        print(
            "Router extraction error: "
            f"{type(exc).__name__}: {exc}"
        )
        data = {
            key: None
            for key in TravelExtraction.model_fields.keys()
        }
        data["intent"] = "general"
        data["sub_action"] = "general"
    
    data["session_id"] = session_id
    
    # ✅ Explicitly extract hotel name from message if not detected
    if data.get("intent") == "hotel" and not data.get("hotel_name"):
        import re
        # Look for hotel name patterns
        hotel_patterns = [
            r'book\s+([\w\s]+?)(?:\s+in\s+\w+)?$',  # "Book Four Seasons BOM 1"
            r'book\s+([\w\s]+?)(?:\s+hotel)?\s+in',  # "Book Four Seasons BOM 1 in Mumbai"
            r'([\w\s]+?)\s+hotel',  # "Four Seasons BOM 1 hotel"
        ]
        for pattern in hotel_patterns:
            match = re.search(pattern, user_msg, re.IGNORECASE)
            if match:
                hotel_name = match.group(1).strip()
                if len(hotel_name) > 3:  # Avoid matching single words
                    data["hotel_name"] = hotel_name
                    print(f"🔍 Extracted hotel name from message: {hotel_name}")
                    break
    
    # Explicit booking detection
    user_msg_lower = user_msg.lower()
    booking_keywords = ['book', 'reserve', 'confirm', 'booking', 'reservation']
    
    if any(keyword in user_msg_lower for keyword in booking_keywords) and data.get("intent") in {"hotel", "flight"}:
        data["sub_action"] = "book"
    
    if data.get("sub_action") == "book" and data.get("intent") == "flight":
        import re
        flight_pattern = re.search(r'\b([A-Z]{2,3}\d{1,4})\b', user_msg.upper())
        if flight_pattern and not data.get("flight_number") and not data.get("flight_id"):
            flight_number = flight_pattern.group(1)
            if re.match(r'^[A-Z]{2,3}\d{1,4}$', flight_number):
                data["flight_number"] = flight_number
                print(f"🔍 Extracted flight number from message: {flight_number}")
    
    return {**data, "hotel_results": [], "flight_results": [], "response_text": "", "session_id": session_id}

# ============================================================
# Helpers
# ============================================================

def _service_error(result, default_msg: str) -> Optional[dict]:
    if isinstance(result, dict) and result.get("ok") is False:
        return {"hotel_results": [], "flight_results": [], "response_text": result.get("error", default_msg)}
    return None

def _extract_items(result, key: str) -> list:
    if isinstance(result, dict):
        return result.get(key, [])
    return result if isinstance(result, list) else []

def _get_id(item: dict, *keys) -> Optional[str]:
    for key in keys:
        if item.get(key):
            return item.get(key)
    return None

def _get_str(item: dict, *keys, default: str = "") -> str:
    for key in keys:
        if item.get(key):
            return str(item.get(key)).strip()
    return default

# ============================================================
# Hotel Helpers 
# ============================================================

def _hotel_id(hotel: dict) -> Optional[str]:
    return _get_id(hotel, "_id", "id", "hotelId")

def _hotel_name(hotel: dict) -> str:
    # Try multiple possible field names
    name = _get_str(hotel, "name", "hotelName", "hotel_name", default=None)
    if name:
        return name
    return "Unknown hotel"

def _hotel_city(hotel: dict) -> str:
    city = hotel.get("city", hotel.get("location", {}).get("city", "Unknown city"))
    if isinstance(city, dict):
        return _get_str(city, "name", "city", default="Unknown city")
    return str(city) if city else "Unknown city"

def _format_hotel(hotel: dict) -> str:
    name = _hotel_name(hotel)
    city = _hotel_city(hotel)
    rating = hotel.get("starRating") or hotel.get("stars") or hotel.get("rating") or "Not available"
    price = hotel.get("pricePerNight") or hotel.get("price") or "Not available"
    currency = hotel.get("currency", "USD")
    rooms = hotel.get("availableRooms") or hotel.get("available_rooms") or hotel.get("available") or "Not available"
    
    # ✅ No hotel ID shown in frontend
    return f"### 🏨 {name}\n📍 **City:** {city}\n⭐ **Rating:** {rating} stars\n💰 **Price:** {currency} {price} per night\n🛏️ **Available rooms:** {rooms}\nTo book, type: `Book {name}`"

def _resolve_hotel(hotel_id: Optional[str], hotel_name: Optional[str], city: Optional[str]) -> dict:
    print(f"🔍 Resolving hotel - hotel_id: {hotel_id}, hotel_name: {hotel_name}, city: {city}")
    
    # ✅ Don't treat hotel_id as a name if it looks like an ID
    if hotel_id:
        import re
        # Check if it looks like an ID (contains underscores or is longer than typical names)
        if "_" in str(hotel_id) or len(str(hotel_id)) > 20 or re.match(r'^[a-z0-9]{20,}$', str(hotel_id)):
            print(f"🔍 Using hotel ID directly: {hotel_id}")
            return {"status": "resolved", "hotel_id": hotel_id, "hotel": None}
        else:
            # It might be a hotel name, treat it as such
            hotel_name = str(hotel_id)
            hotel_id = None
            print(f"🔍 Treating {hotel_name} as hotel name")
    
    # Search for hotels
    result = search_hotel.invoke({"city": city}) if city else get_hotels.invoke({})
    print(f"🔍 Search result type: {type(result)}")
    
    if isinstance(result, dict) and result.get("ok") is False:
        return {"status": "error", "message": result.get("error", "Hotel service unavailable.")}
    
    hotels = _extract_items(result, "hotels")
    if not hotels:
        return {"status": "not_found", "message": "I could not retrieve any hotel options."}
    
    print(f"🔍 Found {len(hotels)} hotels total")
    
    # Log first few hotels for debugging
    for i, h in enumerate(hotels[:3]):
        print(f"  - Hotel {i+1}: {_hotel_name(h)} in {_hotel_city(h)}")
    
    matches = hotels
    
    # ✅ If we have a hotel_name, try to match it
    if hotel_name:
        expected = hotel_name.strip().lower()
        matches = []
        for h in hotels:
            h_name = _hotel_name(h).lower()
            # Check for partial match or exact match
            if expected in h_name or h_name in expected:
                matches.append(h)
                print(f"🔍 Matched: {_hotel_name(h)}")
        print(f"🔍 Found {len(matches)} matches for hotel name: {hotel_name}")
    
    # ✅ If no matches and we have a city, try to match by city
    if not matches and city:
        expected_city = city.strip().lower()
        matches = []
        for h in hotels:
            h_city = _hotel_city(h).lower()
            if expected_city in h_city or h_city in expected_city:
                matches.append(h)
        print(f"🔍 Found {len(matches)} matches for city: {city}")
    
    if not matches:
        # Show some available hotels as examples
        sample_hotels = hotels[:5]
        sample_text = "\n".join(f"- **{_hotel_name(h)}** — {_hotel_city(h)} (⭐ {h.get('starRating', 'N/A')})" for h in sample_hotels)
        return {"status": "not_found", "message": f"I could not find **{hotel_name or 'that hotel'}**.\n\nHere are some available hotels:\n\n{sample_text}\n\nPlease try booking with the exact hotel name."}
    
    # ✅ If multiple matches, show them with names only (no IDs)
    if len(matches) > 1:
        options = "\n".join(
            f"- **{_hotel_name(h)}** — {_hotel_city(h)} (⭐ {h.get('starRating', 'N/A')})"
            for h in matches[:8]
        )
        return {"status": "ambiguous", "message": f"### Which hotel would you like to book?\n\n{options}\n\nPlease reply using the exact hotel name."}
    
    selected = matches[0]
    resolved_id = _hotel_id(selected)
    if not resolved_id:
        return {"status": "error", "message": "The selected hotel does not contain a valid booking ID."}
    
    print(f"✅ Resolved hotel: {_hotel_name(selected)} with ID: {resolved_id}")
    
    return {"status": "resolved", "hotel_id": resolved_id, "hotel": selected}

# ============================================================
# Flight Helpers 
# ============================================================

def _flight_id(flight: dict) -> Optional[str]:
    return _get_id(flight, "_id", "id", "flightId")

def _flight_number(flight: dict) -> str:
    airline = flight.get("airline", "")
    aircraft = flight.get("aircraft", "")
    if not flight.get("flightNumber") and not flight.get("flight_number"):
        airline_code = ''.join([word[0] for word in airline.split()[:2]]).upper() if airline else "FL"
        aircraft_code = aircraft[:2].upper() if aircraft else "00"
        return f"{airline_code}{aircraft_code}01"
    return _get_str(flight, "flightNumber", "flight_number", "flightNo")

def _flight_date(flight: dict) -> str:
    date = _get_str(flight, "flightDate", "date", "departure_date")
    if not date and "_creationTime" in flight:
        import datetime
        try:
            timestamp = flight["_creationTime"] / 1000
            date_obj = datetime.datetime.fromtimestamp(timestamp)
            return date_obj.strftime("%Y-%m-%d")
        except:
            pass
    return date or "Unknown date"

def _flight_origin(flight: dict) -> str:
    origin = flight.get("origin", "")
    if isinstance(origin, dict):
        airport = origin.get("airport", "")
        city = origin.get("city", "")
        return f"{city} ({airport})" if airport and city else city or airport or "Unknown"
    return str(origin) if origin else "Unknown"

def _flight_destination(flight: dict) -> str:
    dest = flight.get("destination", "")
    if isinstance(dest, dict):
        airport = dest.get("airport", "")
        city = dest.get("city", "")
        return f"{city} ({airport})" if airport and city else city or airport or "Unknown"
    return str(dest) if dest else "Unknown"

def _format_flight(flight: dict) -> str:
    airline = flight.get("airline", "Unknown airline")
    aircraft = flight.get("aircraft", "")
    number = _flight_number(flight) or "N/A"
    date = _flight_date(flight) or "Unknown"
    origin = _flight_origin(flight) or "Unknown"
    dest = _flight_destination(flight) or "Unknown"
    departure = flight.get("departureTime") or flight.get("departure_time") or "N/A"
    arrival = flight.get("arrivalTime") or flight.get("arrival_time") or "N/A"
    price = flight.get("price", "N/A")
    currency = flight.get("currency", "USD")
    seats = flight.get("availableSeats") or flight.get("available_seats") or flight.get("seats") or "N/A"
    
    return f"### ✈️ {airline} {number}\n🛫 **Route:** {origin} → {dest}\n📅 **Date:** {date}\n🕒 **Time:** {departure} – {arrival}\n💰 **Price:** {currency} {price}\n💺 **Available seats:** {seats}\nTo book, type: `Book flight {number}`"

def _resolve_flight(flight_id: Optional[str], flight_number: Optional[str], airline: Optional[str], 
                   origin: Optional[str], destination: Optional[str], flight_date: Optional[str]) -> dict:
    if flight_id:
        import re
        is_flight_number_pattern = bool(re.match(r'^[A-Z]{2,3}\d{1,4}$', str(flight_id).upper()))
        
        if is_flight_number_pattern:
            flight_number = str(flight_id).upper()
            flight_id = None
            print(f"🔍 Treating {flight_number} as flight number")
        else:
            print(f"🔍 Using flight ID directly: {flight_id}")
            return {"status": "resolved", "flight_id": flight_id, "flight": None}
    
    result = get_flights.invoke({})
    print(f"🔍 Retrieved flights from MCP")
    
    if isinstance(result, dict) and result.get("ok") is False:
        return {"status": "error", "message": result.get("error", "Flight service unavailable.")}
    
    flights = _extract_items(result, "flights")
    if not flights:
        return {"status": "not_found", "message": "No flights could be retrieved."}
    
    print(f"🔍 Found {len(flights)} total flights")
    
    matches = flights
    
    if flight_number:
        expected = str(flight_number).strip().upper()
        matches = []
        for f in flights:
            f_num = _flight_number(f).upper()
            if expected in f_num or f_num in expected:
                matches.append(f)
            elif f.get("airline") and expected in f.get("airline", "").upper():
                matches.append(f)
            elif f.get("aircraft") and expected in f.get("aircraft", "").upper():
                matches.append(f)
        print(f"🔍 Found {len(matches)} matches for flight number {flight_number}")
    
    if not matches and airline:
        expected = airline.strip().lower()
        matches = [f for f in flights if expected in str(f.get("airline", "")).lower()]
        print(f"🔍 Found {len(matches)} matches for airline {airline}")
    
    if not matches and origin and destination:
        origin_expected = origin.strip().upper()
        dest_expected = destination.strip().upper()
        matches = []
        for f in flights:
            f_origin = _flight_origin(f).upper()
            f_dest = _flight_destination(f).upper()
            if (origin_expected in f_origin or f_origin in origin_expected) and \
               (dest_expected in f_dest or f_dest in dest_expected):
                matches.append(f)
        print(f"🔍 Found {len(matches)} matches for route {origin} → {destination}")
    
    if not matches:
        sample_flights = flights[:3]
        sample_text = "\n".join(f"- {f.get('airline', 'Unknown')} {_flight_number(f)}: {_flight_origin(f)} → {_flight_destination(f)}" for f in sample_flights)
        return {"status": "not_found", "message": f"I could not find flight **{flight_number or airline or flight_id or ''}**.\n\nHere are some available flights:\n\n{sample_text}\n\nPlease try booking with one of these flight numbers."}
    
    if len(matches) > 1:
        options = "\n".join(
            f"- **{f.get('airline', 'Unknown')} {_flight_number(f)}** — {_flight_origin(f)} → {_flight_destination(f)}, {_flight_date(f)}"
            for f in matches[:8]
        )
        return {"status": "ambiguous", "message": f"### Multiple flights found\n\nI found these matching flights:\n\n{options}\n\nPlease specify which one you want to book (use the flight number)."}
    
    selected = matches[0]
    resolved_id = _flight_id(selected)
    if not resolved_id:
        return {"status": "error", "message": "The selected flight does not contain a booking ID."}
    
    print(f"✅ Resolved flight: {selected.get('airline')} with ID: {resolved_id}")
    
    return {"status": "resolved", "flight_id": resolved_id, "flight": selected}

# ============================================================
# Hotel Node 
# ============================================================

def hotel_node(state: GraphState) -> dict:
    session_id = state.get("session_id")
    city, check_in, check_out = state.get("city"), state.get("check_in"), state.get("check_out")
    hotel_id, hotel_name, guest_name, guest_email = state.get("hotel_id"), state.get("hotel_name"), state.get("guest_name"), state.get("guest_email")
    
    print(f"🏨 Hotel Node - sub_action: {state.get('sub_action')}")
    print(f"🏨 Hotel Node - hotel_name: {hotel_name}")
    print(f"🏨 Hotel Node - guest_name: {guest_name}")
    print(f"🏨 Hotel Node - guest_email: {guest_email}")
    
    if state.get("sub_action") == "book":
        resolution = _resolve_hotel(hotel_id, hotel_name, city)
        print(f"🏨 Resolution status: {resolution.get('status')}")
        
        if resolution["status"] != "resolved":
            return {"hotel_results": [], "flight_results": [], "response_text": resolution["message"]}
        
        # Get the hotel name from the resolution
        selected_hotel = resolution.get("hotel") or {}
        hotel_display_name = _hotel_name(selected_hotel) or hotel_name or "Selected Hotel"
        
        missing = [f for f, v in [("check-in date", check_in), ("check-out date", check_out), ("guest name", guest_name), ("guest email", guest_email)] if not v]
        if missing:
            return {"hotel_results": [], "flight_results": [], "response_text": f"### Hotel booking details required\n\nYou selected **{hotel_display_name}**.\n\nPlease provide:\n\n" + "\n".join(f"- {f}" for f in missing) + "\n\nExample:\n\n`2026-07-20 to 2026-07-22, Aathif Aslam, aathif858@gmail.com`"}
        
        print(f"🏨 Calling book_hotel for {resolution['hotel_id']}...")
        
        try:
            result = book_hotel.invoke({
                "hotel_id": resolution["hotel_id"], 
                "guest_name": guest_name, 
                "guest_email": guest_email, 
                "check_in_date": check_in, 
                "check_out_date": check_out, 
                "room_type": "standard"
            })
            
            print(f"🏨 Hotel booking result: {result}")
            
            error = _service_error(result, "Hotel booking service unavailable.")
            if error:
                return error
            
            # Get the actual hotel name from the selected hotel
            hotel_name_final = _hotel_name(selected_hotel) or hotel_name or "Hotel"
            hotel_city = _hotel_city(selected_hotel) or city or "Unknown"
            
            # ✅ Return booking_record like flights do
            return {
                "hotel_results": [], 
                "flight_results": [], 
                "response_text": f"## ✅ Hotel booking confirmed\n\n**{hotel_name_final}** booked for {guest_name} from {check_in} to {check_out}.",
                "booking_record": {
                    "booking_type": "hotel",
                    "status": "confirmed",
                    "provider_id": resolution["hotel_id"],
                    "public_reference": hotel_name_final,
                    "customer_name": guest_name,
                    "customer_email": guest_email,
                    "confirmation_reference": result.get("booking", {}).get("bookingId") or result.get("bookingId") if isinstance(result, dict) else None,
                    "details": {
                        "hotel_name": hotel_name_final,
                        "city": hotel_city,
                        "check_in": check_in,
                        "check_out": check_out,
                        "room_type": "standard",
                        "star_rating": selected_hotel.get("starRating") if selected_hotel else None,
                    }
                }
            }
            
        except Exception as e:
            print(f"❌ Error during hotel booking: {e}")
            import traceback
            traceback.print_exc()
            return {"hotel_results": [], "flight_results": [], 
                    "response_text": f"### Booking Error\n\nThere was an error processing your booking: {str(e)}\n\nPlease try again or contact support."}
    
    params = {"city": city} if city else {}
    if city and check_in:
        params["checkIn"] = check_in
    if city and check_out:
        params["checkOut"] = check_out
    
    result = search_hotel.invoke(params) if city else get_hotels.invoke({})
    error = _service_error(result, "Hotel service unavailable.")
    if error:
        return error
    
    hotels = _extract_items(result, "hotels")
    if not hotels:
        return {"hotel_results": [], "flight_results": [], "response_text": "### No hotels found\n\nHotel searches must use a **city name**, not a country.\n\nExamples:\n\n- `Find hotels in Colombo`\n- `Find hotels in Mumbai`\n- `Find hotels in Bangkok`"}
    
    return {"hotel_results": hotels, "flight_results": [], "response_text": ""}

# ============================================================
# Flight Node
# ============================================================

def flight_node(state: GraphState) -> dict:
    print(f"🔍 Flight Node - sub_action: {state.get('sub_action')}")
    print(f"🔍 Flight Node - flight_id: {state.get('flight_id')}")
    print(f"🔍 Flight Node - flight_number: {state.get('flight_number')}")
    print(f"🔍 Flight Node - passenger_name: {state.get('passenger_name')}")
    print(f"🔍 Flight Node - passenger_email: {state.get('passenger_email')}")
    
    session_id = state.get("session_id")
    origin, destination, flight_date = state.get("origin"), state.get("destination"), state.get("flight_date")
    flight_id, flight_number, airline = state.get("flight_id"), state.get("flight_number"), state.get("airline")
    passenger_name, passenger_email = state.get("passenger_name"), state.get("passenger_email")
    
    # If flight_id looks like a flight number, treat it as flight_number
    if flight_id and not flight_number:
        import re
        if re.match(r'^[A-Z]{2,3}\d{1,4}$', str(flight_id).upper()):
            flight_number = str(flight_id).upper()
            flight_id = None
            print(f"🔍 Converted flight_id to flight_number: {flight_number}")
    
    if state.get("sub_action") == "book":
        print(f"📚 Attempting to book flight...")
        print(f"📚 flight_id: {flight_id}, flight_number: {flight_number}, airline: {airline}")
        
        if not any([flight_id, flight_number, airline, origin, destination]):
            return {"hotel_results": [], "flight_results": [], 
                    "response_text": "### Which flight would you like to book?\n\nUse the visible flight number or route.\n\nExample: `Book flight CE2605` or `Book flight from BOM to DEL`"}
        
        resolution = _resolve_flight(flight_id, flight_number, airline, origin, destination, flight_date)
        
        print(f"📚 Resolution status: {resolution.get('status')}")
        
        if resolution["status"] != "resolved":
            missing = [f for f, v in [("passenger name", passenger_name), ("passenger email", passenger_email)] if not v]
            extra = "\n\nAlso provide:\n\n" + "\n".join(f"- {f}" for f in missing) if missing else ""
            return {"hotel_results": [], "flight_results": [], "response_text": resolution["message"] + extra + "\n\nExample reply:\n\n`Aathif Aslam, aathif858@gmail.com`"}
        
        missing = [f for f, v in [("passenger name", passenger_name), ("passenger email", passenger_email)] if not v]
        if missing:
            selected = resolution.get("flight") or {}
            flight_display = selected.get('airline', 'Flight') if selected else 'Flight'
            number_display = _flight_number(selected) if selected else flight_number
            date_display = _flight_date(selected) if selected else "selected date"
            return {"hotel_results": [], "flight_results": [], 
                    "response_text": f"### Passenger details required\n\nYou selected **{flight_display} {number_display}** on **{date_display}**.\n\nPlease provide:\n\n" + "\n".join(f"- {f}" for f in missing) + "\n\nExample:\n\n`Aathif Aslam, aathif858@gmail.com`"}
        
        print(f"✈️ Booking flight {resolution['flight_id']} for {passenger_name}")
        
        try:
            result = book_flight.invoke({
                "flight_id": resolution["flight_id"],
                "passenger_name": passenger_name,
                "passenger_email": passenger_email,
            })
            
            print(f"📊 Booking result: {result}")
            
            error = _service_error(result, "Flight booking service unavailable.")
            if error:
                return error
            
            # ✅ Get flight details for the confirmation message
            selected_flight = resolution.get("flight") or {}
            flight_airline = selected_flight.get("airline", airline or "Unknown Airline")
            flight_number_display = _flight_number(selected_flight) or flight_number or "N/A"
            flight_origin = _flight_origin(selected_flight) or origin or "Unknown"
            flight_dest = _flight_destination(selected_flight) or destination or "Unknown"
            flight_date_display = _flight_date(selected_flight) or flight_date or "Unknown date"
            flight_price = selected_flight.get("price", "N/A")
            flight_currency = selected_flight.get("currency", "USD")
            
            # ✅ Create detailed confirmation message like hotel
            confirmation_text = f"""## ✅ Flight booking confirmed

**{flight_airline} {flight_number_display}** booked for {passenger_name}

- **Route:** {flight_origin} → {flight_dest}
- **Date:** {flight_date_display}
- **Price:** {flight_currency} {flight_price}
- **Passenger:** {passenger_name}
- **Email:** {passenger_email}"""
            
            # Add booking reference if available
            booking_ref = result.get("booking_id") or result.get("bookingId") or result.get("confirmationId") if isinstance(result, dict) else None
            if booking_ref:
                confirmation_text += f"\n- **Booking Reference:** `{booking_ref}`"
            
            return {
                "hotel_results": [], 
                "flight_results": [], 
                "response_text": confirmation_text,
                "booking_record": {
                    "booking_type": "flight",
                    "status": "confirmed",
                    "provider_id": resolution["flight_id"],
                    "public_reference": flight_number_display,
                    "customer_name": passenger_name,
                    "customer_email": passenger_email,
                    "confirmation_reference": booking_ref,
                    "details": {
                        "airline": flight_airline,
                        "flight_number": flight_number_display,
                        "origin": flight_origin,
                        "destination": flight_dest,
                        "flight_date": flight_date_display,
                        "price": flight_price,
                        "currency": flight_currency,
                    }
                }
            }
            
        except Exception as e:
            print(f"❌ Error during flight booking: {e}")
            import traceback
            traceback.print_exc()
            return {"hotel_results": [], "flight_results": [], 
                    "response_text": f"### Booking Error\n\nThere was an error processing your booking: {str(e)}\n\nPlease try again or contact support."}
    
    if origin and destination:
        params = {"origin": origin, "destination": destination}
        if flight_date:
            params["date"] = flight_date
        result = search_flights.invoke(params)
    elif origin or destination:
        return {"hotel_results": [], "flight_results": [], "response_text": "### Route information required\n\nPlease provide both the origin and destination.\n\nExample: `Find flights from BOM to DEL`"}
    else:
        result = get_flights.invoke({})
    
    error = _service_error(result, "Flight service unavailable.")
    if error:
        return error
    
    flights = _extract_items(result, "flights")
    if flight_date:
        flights = [f for f in flights if _flight_date(f) == flight_date]
    
    if not flights:
        return {"hotel_results": [], "flight_results": [], "response_text": "### No flights found\n\nTry another route or travel date."}
    
    return {"hotel_results": [], "flight_results": flights, "response_text": ""}

# ============================================================
# General Travel Node
# ============================================================

def general_qa_node(state: GraphState) -> dict:
    """Answer general, non-transactional travel questions."""
    messages = state["messages"]
    user_msg = messages[-1]
    history = messages[:-1]

    sys_prompt = get_system_prompt_for_general_qa("\n".join(history))
    inv_msgs = [SystemMessage(content=sys_prompt)]

    for i in range(0, len(history), 2):
        inv_msgs.append(HumanMessage(content=history[i]))
        if i + 1 < len(history):
            inv_msgs.append(AIMessage(content=history[i + 1]))

    inv_msgs.append(HumanMessage(content=user_msg))

    try:
        response = llm.invoke(inv_msgs)
        content = getattr(response, "content", "")
        if not content:
            content = (
                "I could not prepare travel guidance for that request. "
                "Please try rephrasing it."
            )
        return {
            "hotel_results": [],
            "flight_results": [],
            "response_text": content,
        }
    except Exception as exc:
        print(f"General QA error: {type(exc).__name__}: {exc}")
        return {
            "hotel_results": [],
            "flight_results": [],
            "response_text": (
                "I could not generate travel advice right now. "
                "Hotel and flight services are still available."
            ),
        }


# Temporary alias for older imports.
unknown_node = general_qa_node

# ============================================================
# Final Response & Routing
# ============================================================

def generate_response(state: GraphState) -> dict:
    if state.get("response_text"):
        return {"response_text": state["response_text"]}
    
    hotels, flights = state.get("hotel_results", []), state.get("flight_results", [])
    
    if hotels:
        displayed = hotels
        text = "\n\n---\n\n".join(_format_hotel(h) for h in displayed)
        return {"response_text": f"## Hotel options\n\n{text}"}
    
    if flights:
        displayed = flights
        text = "\n\n---\n\n".join(_format_flight(f) for f in displayed)
        return {"response_text": f"## Flight options\n\n{text}"}
    
    return {"response_text": "No matching travel options were found."}

def route_after_extraction(state: GraphState) -> str:
    """Return the graph branch selected by the router."""
    intent = state.get("intent", "general")

    if intent == "hotel":
        return "hotel"

    if intent == "flight":
        return "flight"

    if intent == "planner":
        return "planner"

    return "general"