import json
import secrets
import sqlite3
import string
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "tripweaver.db"

_REFERENCE_ALPHABET = string.ascii_uppercase + string.digits


@contextmanager
def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def _column_exists(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
) -> bool:
    rows = connection.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    return any(row["name"] == column_name for row in rows)


def _generate_booking_reference(
    connection: sqlite3.Connection,
    booking_type: str,
) -> str:
    """
    Generate a user-facing TripWeaver reference.

    Examples:
        TW-2026-F-A7K2Q9
        TW-2026-H-P4M8TX
    """

    type_code = (
        "F"
        if booking_type.lower() == "flight"
        else "H"
        if booking_type.lower() == "hotel"
        else "B"
    )

    year = datetime.now().year

    for _ in range(20):
        random_part = "".join(
            secrets.choice(_REFERENCE_ALPHABET)
            for _ in range(6)
        )
        reference = f"TW-{year}-{type_code}-{random_part}"

        existing = connection.execute(
            """
            SELECT 1
            FROM bookings
            WHERE booking_reference = ?
            """,
            (reference,),
        ).fetchone()

        if existing is None:
            return reference

    raise RuntimeError(
        "Could not generate a unique booking reference."
    )


def initialize_database() -> None:
    """
    Create tables and migrate older TripWeaver databases.

    Existing records receive a TripWeaver reference during migration.
    Records without a session_id remain hidden from session-specific
    booking history.
    """

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                booking_reference TEXT UNIQUE,
                booking_type TEXT NOT NULL,
                provider_id TEXT,
                public_reference TEXT,
                customer_name TEXT,
                customer_email TEXT,
                status TEXT NOT NULL,
                confirmation_reference TEXT,
                booking_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        if not _column_exists(
            connection,
            "bookings",
            "session_id",
        ):
            connection.execute(
                """
                ALTER TABLE bookings
                ADD COLUMN session_id TEXT
                """
            )

        if not _column_exists(
            connection,
            "bookings",
            "booking_reference",
        ):
            connection.execute(
                """
                ALTER TABLE bookings
                ADD COLUMN booking_reference TEXT
                """
            )

        # Backfill references for old development records.
        rows = connection.execute(
            """
            SELECT id, booking_type
            FROM bookings
            WHERE booking_reference IS NULL
               OR TRIM(booking_reference) = ''
            """
        ).fetchall()

        for row in rows:
            reference = _generate_booking_reference(
                connection,
                row["booking_type"],
            )
            connection.execute(
                """
                UPDATE bookings
                SET booking_reference = ?
                WHERE id = ?
                """,
                (
                    reference,
                    row["id"],
                ),
            )

        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_booking_reference
            ON bookings(booking_reference)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_session
            ON chat_messages(session_id)
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_booking_session
            ON bookings(session_id)
            """
        )


def save_chat_message(
    session_id: str,
    role: str,
    content: str,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (
                session_id,
                role,
                content
            )
            VALUES (?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
            ),
        )


def get_recent_chat_messages(
    session_id: str,
    limit: int = 8,
) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                session_id,
                limit,
            ),
        ).fetchall()

    return [
        {
            "role": row["role"],
            "content": row["content"],
        }
        for row in reversed(rows)
    ]


def get_chat_history(session_id: str) -> list[dict]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content, created_at
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

    return [
        {
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def clear_chat_history(session_id: str) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            DELETE FROM chat_messages
            WHERE session_id = ?
            """,
            (session_id,),
        )


def _row_to_booking(row: sqlite3.Row) -> dict:
    try:
        details = json.loads(
            row["booking_details"] or "{}"
        )
    except json.JSONDecodeError:
        details = {}

    booking = {
        "id": row["id"],
        "booking_reference": row["booking_reference"],
        "booking_type": row["booking_type"],
        "provider_id": row["provider_id"],
        "public_reference": row["public_reference"],
        "customer_name": row["customer_name"],
        "customer_email": row["customer_email"],
        "status": row["status"],
        "confirmation_reference": row[
            "confirmation_reference"
        ],
        "details": details,
        "created_at": row["created_at"],
    }

    if "session_id" in row.keys():
        booking["session_id"] = row["session_id"]

    return booking


def save_booking(
    session_id: str,
    booking_type: str,
    status: str,
    provider_id: Optional[str] = None,
    public_reference: Optional[str] = None,
    customer_name: Optional[str] = None,
    customer_email: Optional[str] = None,
    confirmation_reference: Optional[str] = None,
    booking_details: Optional[dict[str, Any]] = None,
) -> str:
    """
    Save one booking and return its TripWeaver reference.
    """

    if not session_id:
        raise ValueError(
            "session_id is required when saving a booking."
        )

    details_json = json.dumps(
        booking_details or {},
        ensure_ascii=False,
    )

    with get_connection() as connection:
        booking_reference = _generate_booking_reference(
            connection,
            booking_type,
        )

        connection.execute(
            """
            INSERT INTO bookings (
                session_id,
                booking_reference,
                booking_type,
                provider_id,
                public_reference,
                customer_name,
                customer_email,
                status,
                confirmation_reference,
                booking_details
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                booking_reference,
                booking_type,
                provider_id,
                public_reference,
                customer_name,
                customer_email,
                status,
                confirmation_reference,
                details_json,
            ),
        )

    return booking_reference


def get_bookings(session_id: str) -> list[dict]:
    """Return bookings belonging only to the requested session."""

    if not session_id:
        return []

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                booking_reference,
                booking_type,
                provider_id,
                public_reference,
                customer_name,
                customer_email,
                status,
                confirmation_reference,
                booking_details,
                created_at
            FROM bookings
            WHERE session_id = ?
            ORDER BY id DESC
            """,
            (session_id,),
        ).fetchall()

    return [_row_to_booking(row) for row in rows]


def get_booking_by_reference(
    session_id: str,
    booking_reference: str,
) -> Optional[dict]:
    """
    Find a booking only when it belongs to the current session.
    """

    if not session_id or not booking_reference:
        return None

    normalized_reference = booking_reference.strip().upper()

    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                id,
                booking_reference,
                booking_type,
                provider_id,
                public_reference,
                customer_name,
                customer_email,
                status,
                confirmation_reference,
                booking_details,
                created_at
            FROM bookings
            WHERE session_id = ?
              AND UPPER(booking_reference) = ?
            LIMIT 1
            """,
            (
                session_id,
                normalized_reference,
            ),
        ).fetchone()

    return _row_to_booking(row) if row else None


def get_all_bookings() -> list[dict]:
    """Development helper; do not expose through a public endpoint."""

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                session_id,
                booking_reference,
                booking_type,
                provider_id,
                public_reference,
                customer_name,
                customer_email,
                status,
                confirmation_reference,
                booking_details,
                created_at
            FROM bookings
            ORDER BY id DESC
            """
        ).fetchall()

    return [_row_to_booking(row) for row in rows]