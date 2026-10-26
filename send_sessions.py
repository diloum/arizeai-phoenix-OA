"""Send test conversations to the travel agent API to exercise session memory.

Each conversation is one session: the first message starts it, and the session_id
returned by the API is sent with every follow-up message.

Each turn carries "frustrated", the human label assigned before running any
evaluation. It is the ground truth the frustration judge is compared against
(log_human_labels.py sends these labels to Phoenix).

Run from the project folder while the app is running on localhost:8000:
    poetry run python send_sessions.py
Results are saved to session_results.json.
"""

import json
import time
import urllib.request

API_URL = "http://localhost:8000/chat"

CONVERSATIONS = [
    {"id": "happy_path", "turns": [
        {"message": "Hi, I'm planning a trip to Lisbon from 2026-11-10 for 3 nights.", "frustrated": False},
        {"message": "Can you find me a hotel under 100 euros a night?", "frustrated": False},
        {"message": "Great, thanks! What's worth visiting near Alfama?", "frustrated": False},
    ]},
    {"id": "city_not_covered", "turns": [
        {"message": "I need a hotel in Berlin from 2026-11-10 for 2 nights.", "frustrated": False},
        {"message": "Seriously? You can't find a single hotel in Berlin? That's pretty useless.", "frustrated": True},
        {"message": "OK, what about Paris for the same dates?", "frustrated": False},
    ]},
    {"id": "vague_date", "turns": [
        {"message": "Find me a hotel in Paris for next Friday, 2 nights.", "frustrated": False},
        {"message": "I mean Friday 2026-11-06. Why do I have to spell everything out?", "frustrated": True},
    ]},
    {"id": "user_changes_own_plan", "turns": [
        {"message": "Hotels in Marseille from 2026-11-20 for 2 nights, please.", "frustrated": False},
        # changes their own plan: a correction, not a complaint
        {"message": "Actually, make it 3 nights instead of 2.", "frustrated": False},
    ]},
    {"id": "nothing_in_budget", "turns": [
        {"message": "Any hotels in Paris from 2026-11-15 for 2 nights under 50 euros a night?", "frustrated": False},
        {"message": "Nothing under 50? Ugh. This is the third site that tells me that.", "frustrated": True},
        {"message": "OK, what's the cheapest option then?", "frustrated": False},
    ]},
    {"id": "polite_follow_up", "turns": [
        {"message": "Can you suggest things to do in Marseille?", "frustrated": False},
        {"message": "Thanks, that's really helpful! Any hotel near the Vieux-Port from 2026-11-20 for 2 nights?", "frustrated": False},
    ]},
    {"id": "frustrated_repeat", "turns": [
        {"message": "Hotels in Lisbon from 2026-11-10 for 3 nights, under 70 euros a night please.", "frustrated": False},
        {"message": "Can you show me more options?", "frustrated": False},
        {"message": "No, I already told you, UNDER 70 a night. Why is that so hard?", "frustrated": True},
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
