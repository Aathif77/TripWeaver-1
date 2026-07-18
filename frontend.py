import json
import os
import uuid
from pathlib import Path
from typing import Generator, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gradio as gr


BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "tripweaver.css"

API_BASE_URL = os.environ.get(
    "TRAVEL_PLANNER_API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

CHAT_STREAM_URL = f"{API_BASE_URL}/chat/stream"
BOOKINGS_URL = f"{API_BASE_URL}/bookings"
CHAT_HISTORY_URL = f"{API_BASE_URL}/chat-history"


def ensure_session_id(existing_session_id: str) -> str:
    """
    Reuse the browser's saved session ID.

    Generate a new one only when this browser has never used
    TripWeaver before.
    """

    if existing_session_id:
        return existing_session_id

    return str(uuid.uuid4())

def status_html(
    mode: str,
    message: str = "",
) -> str:
    """Return compact status markup used by the unchanged chat header."""

    if mode == "ready":
        return """
        <div class="status-wrap">
            <span class="status-dot"></span>
            <span>Ready</span>
        </div>
        """

    display_message = message or "Planning your journey..."

    return f"""
    <div class="status-wrap">
        <span class="status-spinner"></span>
        <span class="status-copy">
            <strong>{display_message}</strong>
            <small>TripWeaver is working on your request</small>
        </span>
    </div>
    """


def stream_chat_api(
    message: str,
    session_id: str,
) -> Iterator[dict]:
    """Read newline-delimited streaming events from FastAPI."""

    payload = json.dumps(
        {
            "message": message,
            "session_id": session_id,
        }
    ).encode("utf-8")

    request = Request(
        CHAT_STREAM_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()

                if not line:
                    continue

                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    except HTTPError as exc:
        yield {
            "type": "error",
            "message": (
                f"The backend returned status {exc.code}. "
                "Please try again."
            ),
        }

    except URLError:
        yield {
            "type": "error",
            "message": (
                "The backend is unavailable. "
                "Start it with `python main.py`."
            ),
        }

    except TimeoutError:
        yield {
            "type": "error",
            "message": "The request timed out. Please try again.",
        }

    except Exception as exc:
        yield {
            "type": "error",
            "message": f"Unexpected error: {type(exc).__name__}",
        }


def respond(
    message: str,
    history: list | None,
    session_id: str,
) -> Generator[tuple[str, list, str], None, None]:
    """Show activity immediately and update one assistant message in place."""

    history = history or []
    clean_message = (message or "").strip()

    if not clean_message:
        yield "", history, status_html("ready")
        return

    user_history = history + [
        {
            "role": "user",
            "content": clean_message,
        }
    ]

    yield (
        "",
        user_history,
        status_html(
            "loading",
            "Understanding your request...",
        ),
    )

    accumulated = ""
    assistant_added = False

    for event in stream_chat_api(
        message=clean_message,
        session_id=session_id,
    ):
        event_type = event.get("type")

        if event_type == "status":
            yield (
                "",
                (
                    user_history
                    if not assistant_added
                    else user_history
                    + [{
                        "role": "assistant",
                        "content": accumulated,
                    }]
                ),
                status_html(
                    "loading",
                    event.get("message", "Working..."),
                ),
            )

        elif event_type == "token":
            accumulated += event.get("content", "")
            assistant_added = True

            yield (
                "",
                user_history + [
                    {
                        "role": "assistant",
                        "content": accumulated,
                    }
                ],
                status_html(
                    "loading",
                    "Writing your response...",
                ),
            )

        elif event_type == "error":
            error_message = event.get(
                "message",
                "TripWeaver could not complete the request.",
            )

            yield (
                "",
                user_history + [
                    {
                        "role": "assistant",
                        "content": (
                            "### Unable to complete the request\n\n"
                            f"{error_message}"
                        ),
                    }
                ],
                status_html("ready"),
            )
            return

        elif event_type == "done":
            final_response = event.get("response", accumulated)

            yield (
                "",
                user_history + [
                    {
                        "role": "assistant",
                        "content": final_response,
                    }
                ],
                status_html("ready"),
            )
            return

    if accumulated:
        yield (
            "",
            user_history + [
                {
                    "role": "assistant",
                    "content": accumulated,
                }
            ],
            status_html("ready"),
        )


def get_bookings(session_id: str) -> str:
    """Fetch and format booking history."""

    try:
        request = Request(
            f"{BOOKINGS_URL}/{session_id}",
            headers={"Accept": "application/json"},
            method="GET",
        )

        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

    except HTTPError as exc:
        return f"⚠️ Failed to load bookings: {exc.code}"

    except URLError:
        return "⚠️ Backend unavailable. Start it with `python main.py`."

    except Exception as exc:
        return f"⚠️ Booking error: {type(exc).__name__}"

    bookings = data.get("bookings", [])

    if not bookings:
        return "📭 No bookings found for this session."

    output = ["## 📋 Your Bookings\n"]

    for index, booking in enumerate(bookings, 1):
        booking_type = booking.get(
            "booking_type",
            "",
        ).lower()

        details = booking.get("details", {})
        booking_status = booking.get("status", "Unknown")
        reference = (
            booking.get("booking_reference")
            or "Reference unavailable"
        )

        if booking_type == "flight":
            output.append(
                f"### {index}. ✈️ "
                f"{details.get('airline', 'Unknown Airline')} "
                f"{details.get('flight_number', '')}\n"
                f"- **Route:** {details.get('origin', 'N/A')} → "
                f"{details.get('destination', 'N/A')}\n"
                f"- **Date:** {details.get('flight_date', 'N/A')}\n"
                f"- **Status:** {booking_status}\n"
                f"- **TripWeaver Ref:** `{reference}`\n"
                f"- **Passenger:** "
                f"{booking.get('customer_name', 'N/A')}\n"
            )

        elif booking_type == "hotel":
            output.append(
                f"### {index}. 🏨 "
                f"{details.get('hotel_name', 'Hotel')}\n"
                f"- **City:** {details.get('city', 'N/A')}\n"
                f"- **Check-in:** {details.get('check_in', 'N/A')}\n"
                f"- **Check-out:** {details.get('check_out', 'N/A')}\n"
                f"- **Status:** {booking_status}\n"
                f"- **TripWeaver Ref:** `{reference}`\n"
                f"- **Guest:** "
                f"{booking.get('customer_name', 'N/A')}\n"
            )

        output.append("---\n")

    return "\n".join(output)


def show_bookings(
    session_id: str,
    history: list | None,
) -> Generator[tuple[str, list, str], None, None]:
    history = history or []

    user_history = history + [
        {
            "role": "user",
            "content": "📋 Show my bookings",
        }
    ]

    yield (
        "",
        user_history,
        status_html("loading", "Loading your bookings..."),
    )

    bookings_text = get_bookings(session_id)

    yield (
        "",
        user_history + [
            {
                "role": "assistant",
                "content": bookings_text,
            }
        ],
        status_html("ready"),
    )


def clear_chat(session_id: str):
    """Clear the visible chat and the matching backend conversation."""

    try:
        request = Request(
            f"{CHAT_HISTORY_URL}/{session_id}",
            method="DELETE",
        )
        with urlopen(request, timeout=15):
            pass
    except Exception:
        # The visible chat should still clear if backend cleanup fails.
        pass

    return "", [], status_html("ready")


def load_css() -> str:
    if not CSS_PATH.exists():
        raise FileNotFoundError(
            f"CSS file not found: {CSS_PATH}"
        )

    return CSS_PATH.read_text(encoding="utf-8")


def create_ui():
    """Create the existing Gradio UI without changing its structure."""

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="slate",
        radius_size="lg",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(
        css=load_css(),
        theme=theme,
        title="TripWeaver - AI Travel Assistant",
    ) as demo:
        session_id = gr.BrowserState(
            default_value="",
            storage_key="tripweaver_session_id",
        )

        gr.HTML("""
        <div class="slider-container">
            <div class="slider-track">
                <div class="slide" style="background-image: url('https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=1200&q=80');">
                    <div class="slide-content">
                        <h2>✈️ Dream Destinations</h2>
                        <p>Explore the world with AI-powered travel planning</p>
                    </div>
                </div>
                <div class="slide" style="background-image: url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=1200&q=80');">
                    <div class="slide-content">
                        <h2>🏖️ Beach Getaways</h2>
                        <p>Find the perfect tropical paradise</p>
                    </div>
                </div>
                <div class="slide" style="background-image: url('https://images.unsplash.com/photo-1501785888041-af3ef285b470?w=1200&q=80');">
                    <div class="slide-content">
                        <h2>🏔️ Mountain Adventures</h2>
                        <p>Discover breathtaking peaks and valleys</p>
                    </div>
                </div>
                <div class="slide" style="background-image: url('https://images.unsplash.com/photo-1519681393784-d120267933ba?w=1200&q=80');">
                    <div class="slide-content">
                        <h2>🌆 City Explorations</h2>
                        <p>Navigate urban jungles with ease</p>
                    </div>
                </div>
                <div class="slide" style="background-image: url('https://images.unsplash.com/photo-1508672019048-805c876b67e2?w=1200&q=80');">
                    <div class="slide-content">
                        <h2>🌏 Cultural Journeys</h2>
                        <p>Immerse yourself in local experiences</p>
                    </div>
                </div>
            </div>
            <div class="slider-dots">
                <span class="active"></span>
                <span></span><span></span><span></span><span></span>
            </div>
        </div>
        """)

        gr.HTML("""
        <div class="hero-banner">
            <div class="hero-content">
                <h1><span>🧭</span> TripWeaver</h1>
                <p>Your AI travel companion — flights, hotels, and destination insights</p>
            </div>
            <span class="hero-badge">✨ AI-Powered · Free</span>
        </div>
        """)

        with gr.Row(
            elem_id="main-layout",
            equal_height=False,
        ):
            with gr.Column(
                scale=2,
                elem_id="sidebar-col",
            ):
                with gr.Column(elem_classes=["card"]):
                    gr.HTML(
                        '<div class="sidebar-label">'
                        '⚡ Quick Actions</div>'
                    )

                    btn_hotels = gr.Button(
                        "🏨 Hotels in Mumbai",
                        elem_classes=["sidebar-btn"],
                    )
                    btn_flights = gr.Button(
                        "✈️ BOM → DEL Flights",
                        elem_classes=["sidebar-btn"],
                    )
                    btn_advice = gr.Button(
                        "🌏 Thailand Travel Tips",
                        elem_classes=["sidebar-btn"],
                    )
                    btn_bookings = gr.Button(
                        "📋 My Bookings",
                        elem_classes=[
                            "sidebar-btn",
                            "bookings",
                        ],
                    )

                    gr.HTML(
                        '<div class="sidebar-divider"></div>'
                    )
                    gr.HTML("""
                    <div class="sidebar-tip">
                        <strong>💡 Pro Tip</strong>
                        Search by city for best results.<br>
                        Try: <strong>Colombo</strong> ·
                        <strong>Bangkok</strong> ·
                        <strong>Dubai</strong>
                    </div>
                    """)

            with gr.Column(
                scale=10,
                elem_id="chat-col",
            ):
                with gr.Column(elem_id="chat-shell"):
                    with gr.Row(elem_id="chat-header"):
                        gr.HTML("""
                        <div class="assistant-info">
                            <div class="assistant-avatar">🤖</div>
                            <div>
                                <div class="assistant-name">
                                    Travel Assistant
                                </div>
                                <div class="assistant-desc">
                                    Flights · Hotels · Bookings · Advice
                                </div>
                            </div>
                        </div>
                        """)

                        status = gr.HTML(
                            status_html("ready")
                        )

                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        height=550,
                        label=None,
                        show_label=False,
                        container=False,
                    )

                    with gr.Column(elem_id="composer"):
                        with gr.Row(equal_height=True):
                            msg = gr.Textbox(
                                placeholder=(
                                    "Ask about flights, hotels, "
                                    "or destinations..."
                                ),
                                lines=2,
                                max_lines=5,
                                container=False,
                                scale=10,
                                elem_id="msg-input",
                            )

                            send = gr.Button(
                                "Send ✈️",
                                variant="primary",
                                scale=1,
                                elem_id="send-btn",
                            )

                        with gr.Row(
                            elem_classes=["composer-footer"]
                        ):
                            clear = gr.Button(
                                "🗑️ Clear Chat",
                                elem_classes=["clear-btn"],
                            )
                            gr.HTML(
                                '<span class="composer-hint">'
                                'Press Enter to send</span>'
                            )

        send.click(
            respond,
            inputs=[msg, chatbot, session_id],
            outputs=[msg, chatbot, status],
            show_progress="hidden",
        )

        msg.submit(
            respond,
            inputs=[msg, chatbot, session_id],
            outputs=[msg, chatbot, status],
            show_progress="hidden",
        )

        clear.click(
            clear_chat,
            inputs=[session_id],
            outputs=[msg, chatbot, status],
            show_progress="hidden",
        )

        btn_bookings.click(
            show_bookings,
            inputs=[session_id, chatbot],
            outputs=[msg, chatbot, status],
            show_progress="hidden",
        )

        btn_hotels.click(
            lambda: "Find hotels in Mumbai",
            outputs=msg,
        )
        btn_flights.click(
            lambda: (
                "Find flights from BOM to DEL "
                "on 2025-11-15"
            ),
            outputs=msg,
        )
        btn_advice.click(
            lambda: (
                "What should I know before "
                "travelling to Thailand?"
            ),
            outputs=msg,
        )

        demo.load(
            fn=ensure_session_id,
            inputs=[session_id],
            outputs=[session_id],
            show_progress="hidden",
        )

    return demo


def main():
    demo = create_ui()
    demo.launch(
        share=False,
        debug=False,
    )


if __name__ == "__main__":
    main()