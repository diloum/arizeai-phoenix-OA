"""Send test conversations to the travel agent API to exercise session memory.

Each conversation is one session: the first message starts it, and the session_id
returned by the API is sent with every follow-up message.

Run from the project folder while the app is running on localhost:8000:
    poetry run python send_sessions.py
Results are saved to session_results.json.
"""

import json
import time
import urllib.request

API_URL = "http://localhost:8000/chat"

# "check" says what a good answer looks like, so results are easy to review.
CONVERSATIONS = [
    # --- Single-turn sessions ---
    {"id": "single_hotel", "turns": [
        {"message": "Find me a hotel in Tokyo from 2026-12-05 for 4 nights.",
         "check": "Lists Asakusa Capsule (45) and Shinjuku Skyline (220), totals for 4 nights."},
    ]},
    {"id": "single_search", "turns": [
        {"message": "What are the top things to do in Marseille in winter?",
         "check": "Uses web search, no hotel tool."},
    ]},
    {"id": "single_greeting", "turns": [
        {"message": "Hi! What can you help me with?",
         "check": "No tool call."},
    ]},
    {"id": "single_no_budget_match", "turns": [
        {"message": "Hotels in Paris under 50 euros a night, checking in 2026-11-15 for 2 nights.",
         "check": "Says nothing fits the budget; may suggest Montmartre Budget Stay (85)."},
    ]},

    # --- Multi-turn: budget added later ---
    {"id": "budget_correction", "turns": [
        {"message": "I'm heading to Lisbon on 2026-11-10 for 3 nights. Any hotels?",
         "check": "Lists all 3 Lisbon hotels."},
        {"message": "Actually my budget is 100 euros a night max.",
         "check": "Re-runs with max 100: Bairro Alto Rooms and Alfama Courtyard Inn only. Keeps Lisbon and dates."},
        {"message": "Great, what's worth seeing near the cheapest one?",
         "check": "Knows the cheapest is Bairro Alto Rooms; searches the Bairro Alto area."},
    ]},

    # --- Multi-turn: wrong city corrected ---
    {"id": "city_correction", "turns": [
        {"message": "Find me a hotel in Paris from 2026-11-20 for 2 nights.",
         "check": "Lists Paris hotels."},
        {"message": "Sorry, I meant Marseille, not Paris.",
         "check": "Re-runs for Marseille with the same dates (2026-11-20, 2 nights)."},
        {"message": "Which of those is closest to the old port?",
         "check": "Vieux-Port Suites; answers about Marseille, not Paris."},
    ]},

    # --- Multi-turn: number of nights changed ---
    {"id": "nights_correction", "turns": [
        {"message": "I need a hotel in Paris checking in 2026-12-01 for 2 nights, under 200 euros a night.",
         "check": "Montmartre Budget Stay (85) and Le Petit Marais (180)."},
        {"message": "Make that 4 nights instead of 2.",
         "check": "Same city, date and budget; totals now for 4 nights (340 and 720)."},
        {"message": "What would I pay in total for the cheapest one?",
         "check": "340 EUR."},
    ]},

    # --- Multi-turn: trip details given one piece at a time ---
    {"id": "details_over_turns", "turns": [
        {"message": "I want to plan a trip to Tokyo.",
         "check": "Asks for dates/nights, or gives general help."},
        {"message": "I'll arrive on 2026-12-05.",
         "check": "Remembers Tokyo; may ask how many nights."},
        {"message": "Staying 3 nights, and I don't want to spend more than 50 a night.",
         "check": "Calls the hotel tool with Tokyo, 2026-12-05, 3 nights, max 50: Asakusa Capsule & Lounge."},
        {"message": "How do I get from Narita airport to that hotel?",
         "check": "Knows 'that hotel' is in Asakusa; searches Narita to Asakusa."},
    ]},

    # --- Multi-turn: unsupported city, then switch ---
    {"id": "unsupported_city_switch", "turns": [
        {"message": "Find me a hotel in Berlin from 2026-11-10 for 2 nights.",
         "check": "No data for Berlin; lists available cities."},
        {"message": "Okay, Lisbon then, same dates.",
         "check": "Lisbon, 2026-11-10, 2 nights."},
    ]},

    # --- Multi-turn: frustrated traveler repeating themselves ---
    {"id": "frustrated_repeat", "turns": [
        {"message": "Hotels in Lisbon from 2026-11-10 for 3 nights, under 70 euros a night please.",
         "check": "Bairro Alto Rooms only."},
        {"message": "Can you show me more options?",
         "check": "Explains nothing else is under 70; keeps the budget unless asked."},
        {"message": "No, I already told you, UNDER 70 a night. Why is that so hard?",
         "check": "Stays calm, confirms Bairro Alto Rooms is the only match."},
    ]},
]


def send(message: str, session_id: str | None) -> dict:
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(API_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def main() -> None:
    results = []
    for conversation in CONVERSATIONS:
        print(f"=== {conversation['id']}")
        session_id = None  # new session for each conversation
        for turn_number, turn in enumerate(conversation["turns"], start=1):
            print(f"  [{turn_number}] {turn['message']}")
            start = time.time()
            try:
                data, error = send(turn["message"], session_id), None
                session_id = data["session_id"]
                answer = data["response"]
            except Exception as exc:  # keep going if one message fails
                answer, error = None, str(exc)
            seconds = round(time.time() - start, 1)
            print(f"      -> {seconds}s  {(answer or error)[:120]!r}")
            results.append({
                "conversation": conversation["id"], "turn": turn_number, "session_id": session_id,
                **turn, "response": answer, "error": error, "seconds": seconds,
            })
        print()

    with open("session_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to session_results.json")


if __name__ == "__main__":
    main()
