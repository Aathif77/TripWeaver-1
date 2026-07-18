# ✈️ TripWeaver – MCP-Based Multi-Agent Travel Planner

TripWeaver is an AI-powered **Multi-Agent Travel Planning System** developed using **LangGraph**, **FastAPI**, **Gradio**, and the **Model Context Protocol (MCP)**. The application intelligently routes user requests to specialized AI agents that can search flights, search hotels, generate travel itineraries, answer travel-related questions, and manage bookings.

---

# Features

## 🤖 Multi-Agent Architecture

TripWeaver consists of four specialized AI agents:

- **Flight Agent** – Search and book flights
- **Hotel Agent** – Search and book hotels
- **Planner Agent** – Generate complete travel itineraries with budget analysis
- **General QA Agent** – Answer travel-related questions

---

## ✈️ Flight Agent

- Search flights by origin, destination and date
- View airline, flight number, departure/arrival times, price and available seats
- Book flights
- Generate booking confirmation
- Store booking information

---

## 🏨 Hotel Agent

- Search hotels by city
- View hotel rating, nightly price and room availability
- Book hotels
- Generate booking confirmation
- Store booking information

---

## 🗺️ Planner Agent

The Planner Agent combines the Flight and Hotel agents to generate complete travel plans.

Features include:

- Flight recommendations
- Hotel recommendations
- Budget calculation
- Day-by-day itinerary
- Travel tips
- Next-step suggestions

---

## 💬 General Travel Assistant

Provides travel advice including:

- Visa information
- Weather guidance
- Packing suggestions
- Local transportation
- Safety tips
- Destination recommendations

---

## 📚 Booking Management

Supports:

- Flight booking
- Hotel booking
- Booking confirmation
- Booking lookup
- SQLite booking storage

---

# Technology Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| FastAPI | Backend API |
| Gradio | Web Interface |
| LangGraph | Multi-Agent Workflow |
| LangChain | LLM Integration |
| OpenAI GPT | Natural Language Processing |
| SQLite | Booking & Session Storage |
| MCP | Hotel & Flight Services |

---

# Project Structure

```
tripweaver/
│
├── agents/
│   ├── router.py
│   ├── planner.py
│   ├── entity.py
│   ├── llm.py
│   ├── prompts.py
│   ├── tools.py
│   └── workflow.py
│
├── database/
│
├── frontend.py
├── main.py
├── requirements.txt
└── README.md
```

---

# Setup

## 1. Clone the Repository

```bash
git clone <repository-url>
cd tripweaver
```

---

## 2. Create Virtual Environment

```bash
python -m venv env
```

Activate:

### Windows (CMD)

```bash
env\Scripts\activate
```

### Windows (PowerShell)

```powershell
env\Scripts\Activate.ps1
```

### macOS/Linux

```bash
source env/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```text
OPENAI_API_KEY=your_openai_api_key
```

---

# Running the Application

## Start the Backend

```bash
python main.py
```

Backend runs on:

```
http://127.0.0.1:8000
```

---

## Start the Frontend

Open a new terminal:

```bash
python frontend.py
```

Frontend runs on:

```
http://127.0.0.1:7860
```

---

# API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Multi-agent conversation endpoint |
| `/hotels` | GET | Retrieve hotel dataset |
| `/flights` | GET | Retrieve flight dataset |

---

# Example Prompts

## Flight Search

```
Find flights from BOM to DEL on 2025-11-15
```

---

## Hotel Search

```
Find hotels in Mumbai
```

---

## Planner

```
Plan a 5-day trip from Mumbai to Delhi starting on 2025-11-15 for 2 travellers with a budget of USD 1200
```

---

## Flight Booking

```
Book flight CE2605
```

---

## Hotel Booking

```
Book Hyatt BOM 5
```

---

## Booking Lookup

```
Show booking TW-2026-H-L3JFON
```

---

# Error Handling

TripWeaver gracefully handles:

- Backend unavailable
- MCP service unavailable
- Invalid booking references
- Missing travel details
- Invalid routes
- No matching flights
- No matching hotels

---

# Testing

The system has been tested for:

- Flight search
- Hotel search
- Planner functionality
- Flight booking
- Hotel booking
- Booking lookup
- Browser session persistence
- Backend availability
- Invalid input handling

---

# Future Improvements

- Return flight planning
- Pagination for search results
- Weather integration
- Currency conversion
- Interactive maps
- Email confirmations
- Payment gateway integration

---

# License

This project was developed for academic coursework and educational purposes.

---

# Author

**Aathif Aslam**