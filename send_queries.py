"""Send a fixed set of test queries to the travel agent (Step 4).

Each query has the tool we expect the agent to use and whether we (humans) consider
the message frustrated. Those expected labels are our ground truth for checking the
frustration evaluator in Step 5.

Run from the project folder while the app is running on localhost:8000:
    poetry run python send_queries.py
Results are saved to query_results.json.
"""

import json
import time
import urllib.request

API_URL = "http://localhost:8000/chat"

QUERIES = [
    # --- Hotel tool, normal requests ---
    {"id": 1, "category": "hotel", "expected_tool": "get_hotel_options", "expected_frustration": False,
     "message": "Find me a hotel in Lisbon from 2026-11-10 for 3 nights under 100 euros a night."},
    {"id": 2, "category": "hotel", "expected_tool": "get_hotel_options", "expected_frustration": False,
     "message": "I need somewhere to stay in Tokyo, checking in 2026-12-05 for 4 nights."},
    {"id": 3, "category": "hotel", "expected_tool": "get_hotel_options", "expected_frustration": False,
     "message": "What's the cheapest hotel in Marseille for 2 nights starting 2026-11-20?"},

    # --- Hotel tool, edge cases ---
    {"id": 4, "category": "hotel_edge", "expected_tool": "get_hotel_options", "expected_frustration": False,
     "message": "Find me a hotel in Berlin from 2026-11-10 for 2 nights."},  # city not in data
    {"id": 5, "category": "hotel_edge", "expected_tool": "get_hotel_options", "expected_frustration": False,
     "message": "Hotels in Paris under 50 euros a night, checking in 2026-11-15 for 2 nights."},  # nothing in budget
    {"id": 6, "category": "hotel_edge", "expected_tool": "get_hotel_options or none", "expected_frustration": False,
     "message": "Book me a hotel in Paris next Friday for the weekend."},  # vague date

    # --- Web search ---
    {"id": 7, "category": "search", "expected_tool": "duckduckgo_search", "expected_frustration": False,
     "message": "What are the top things to do in Lisbon in November?"},
    {"id": 8, "category": "search", "expected_tool": "duckduckgo_search", "expected_frustration": False,
     "message": "What's the best way to get from Tokyo Narita airport to Shinjuku?"},

    # --- Both tools ---
    {"id": 9, "category": "both", "expected_tool": "both", "expected_frustration": False,
     "message": "I'm going to Lisbon on 2026-11-10 for 3 nights. Suggest a hotel under 120 euros a night and what to see nearby."},

    # --- No tool ---
    {"id": 10, "category": "no_tool", "expected_tool": "none", "expected_frustration": False,
     "message": "Hi! What can you help me with?"},

    # --- Frustrated messages (single-turn, so the cues are inside the message; see send_sessions.py for multi-turn) ---
    {"id": 11, "category": "frustrated", "expected_tool": "get_hotel_options", "expected_frustration": True,
     "message": "This is the third time I'm asking: I need a hotel in Lisbon UNDER 100 euros a night from 2026-11-10 for 3 nights, not a list of attractions."},
    {"id": 12, "category": "frustrated", "expected_tool": "get_hotel_options", "expected_frustration": True,
     "message": "That's wrong, I said Marseille, not Paris. Hotels in Marseille from 2026-11-20 for 2 nights."},
    {"id": 13, "category": "frustrated", "expected_tool": "get_hotel_options", "expected_frustration": True,
     "message": "Why do you keep searching the web? Just give me hotel options in Tokyo from 2026-12-05 for 3 nights."},
    {"id": 14, "category": "frustrated", "expected_tool": "none", "expected_frustration": True,
     "message": "Ugh, forget it. Is there anything in Paris for less than 90 a night or not?"},

    # --- Neutral controls that look similar but are not frustrated ---
    {"id": 15, "category": "neutral_control", "expected_tool": "get_hotel_options", "expected_frustration": False,
     "message": "Thanks! Could you also check hotels in Paris from 2026-12-01 for 2 nights?"},
    {"id": 16, "category": "neutral_control", "expected_tool": "get_hotel_options", "expected_frustration": False,
     "message": "Actually, make that 4 nights in Paris from 2026-12-01 instead of 2."},  # polite correction
]


def send(message: str) -> str:
    body = json.dumps({"message": message}).encode("utf-8")
    request = urllib.request.Request(API_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())["response"]


def main() -> None:
    results = []
    for query in QUERIES:
        print(f"[{query['id']:>2}] {query['message']}")
        start = time.time()
        try:
            answer, error = send(query["message"]), None
        except Exception as exc:  # keep going if one query fails
            answer, error = None, str(exc)
        seconds = round(time.time() - start, 1)
        print(f"     -> {seconds}s  {(answer or error)[:120]!r}\n")
        results.append({**query, "response": answer, "error": error, "seconds": seconds})

    with open("query_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to query_results.json")


if __name__ == "__main__":
    main()
