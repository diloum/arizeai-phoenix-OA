"""Hotel search tool for the travel agent.

Uses a small, fixed (mock) inventory so results are repeatable for demos and evals.
In production this function would call a real hotel/booking API instead.
"""

from datetime import date
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# --- Mock inventory: prices are per night, in EUR -------------------------------
HOTELS = {
    "lisbon": [
        {"name": "Alfama Courtyard Inn", "neighborhood": "Alfama", "stars": 3, "price_per_night": 95},
        {"name": "Hotel Avenida Grand", "neighborhood": "Avenida da Liberdade", "stars": 5, "price_per_night": 310},
        {"name": "Bairro Alto Rooms", "neighborhood": "Bairro Alto", "stars": 2, "price_per_night": 60},
    ],
    "paris": [
        {"name": "Le Petit Marais", "neighborhood": "Le Marais", "stars": 3, "price_per_night": 180},
        {"name": "Hotel Rive Gauche", "neighborhood": "Saint-Germain", "stars": 4, "price_per_night": 260},
        {"name": "Montmartre Budget Stay", "neighborhood": "Montmartre", "stars": 2, "price_per_night": 85},
    ],
    "marseille": [
        {"name": "Vieux-Port Suites", "neighborhood": "Vieux-Port", "stars": 4, "price_per_night": 170},
        {"name": "Le Panier Guesthouse", "neighborhood": "Le Panier", "stars": 2, "price_per_night": 70},
    ],
    "tokyo": [
        {"name": "Shinjuku Skyline Hotel", "neighborhood": "Shinjuku", "stars": 4, "price_per_night": 220},
        {"name": "Asakusa Capsule & Lounge", "neighborhood": "Asakusa", "stars": 2, "price_per_night": 45},
    ],
}


# --- Input schema: what the LLM must provide ---------------------------------------
class HotelSearchInput(BaseModel):
    city: str = Field(description="City to search, e.g. 'Lisbon'.")
    check_in: str = Field(description="Check-in date in YYYY-MM-DD format.")
    nights: int = Field(description="Number of nights, at least 1.")
    max_price_per_night: Optional[float] = Field(
        default=None, description="Optional budget cap per night, in EUR."
    )


# --- Output schema: the structured result the tool always returns ---------------------
class HotelOption(BaseModel):
    name: str
    neighborhood: str
    stars: int
    price_per_night: float
    total_price: float


class HotelSearchResult(BaseModel):
    status: str  # "ok", "no_results" or "error"
    city: str
    check_in: str
    nights: int
    currency: str = "EUR"
    options: list[HotelOption] = []
    message: Optional[str] = None


@tool("get_hotel_options", args_schema=HotelSearchInput)
def get_hotel_options(
    city: str, check_in: str, nights: int, max_price_per_night: Optional[float] = None
) -> str:
    """Find hotel options in a city for given dates and an optional nightly budget.

    Use this whenever the user asks where to stay, for hotel or accommodation
    recommendations, or for lodging prices. Do NOT use web search for hotels.
    """

    def result(**kwargs) -> str:
        return HotelSearchResult(city=city, check_in=check_in, nights=nights, **kwargs).model_dump_json()

    # Validate inputs and return a structured error instead of crashing
    try:
        date.fromisoformat(check_in)
    except ValueError:
        return result(status="error", message="check_in must be a date in YYYY-MM-DD format.")
    if nights < 1:
        return result(status="error", message="nights must be at least 1.")

    hotels = HOTELS.get(city.strip().lower())
    if hotels is None:
        return result(
            status="no_results",
            message=f"No hotel data for {city}. Available cities: {', '.join(c.title() for c in HOTELS)}.",
        )

    matches = [
        HotelOption(**h, total_price=h["price_per_night"] * nights)
        for h in hotels
        if max_price_per_night is None or h["price_per_night"] <= max_price_per_night
    ]
    if not matches:
        return result(status="no_results", message="No hotels within that budget.")

    matches.sort(key=lambda h: h.price_per_night)
    return result(status="ok", options=matches)
