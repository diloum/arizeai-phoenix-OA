"""Attach the human frustration labels from send_sessions.py to their spans and sessions in Phoenix.

Each /chat turn is one trace; the label goes on its root span (the LangGraph span),
found by session_id + user message. Each conversation also gets a session label:
frustrated if any of its turns is. Logged with annotator_kind="HUMAN", so the
labels show in the Phoenix UI next to the LLM judge's results.

Re-running is safe: Phoenix updates an existing annotation with the same name
on the same span or session instead of adding a duplicate.

Run from the project folder after send_sessions.py, with Phoenix running:
    poetry run python log_human_labels.py
"""

import json

import pandas as pd
from phoenix.client import Client

PROJECT_NAME = "travel-agent-arize-OA"
ANNOTATION_NAME = "human_frustration"


def user_message(input_value: str) -> str:
    """The user message sent in this turn, from the root span's input."""
    return json.loads(input_value)["messages"][-1]["data"]["content"]


def main() -> None:
    client = Client()
    results = json.load(open("session_results.json"))

    spans = client.spans.get_spans_dataframe(project_name=PROJECT_NAME, root_spans_only=True, limit=5000)
    spans = spans[spans["attributes.session.id"].isin({r["session_id"] for r in results})]
    span_ids = {
        (row["attributes.session.id"], user_message(row["attributes.input.value"])): span_id
        for span_id, row in spans.iterrows()
    }

    rows = []
    for r in results:
        span_id = span_ids.get((r["session_id"], r["message"]))
        if span_id is None:
            print(f"No span found for {r['conversation']} turn {r['turn']}, skipped")
            continue
        rows.append({
            "span_id": span_id,
            "label": "frustrated" if r["frustrated"] else "not_frustrated",
            "score": 1.0 if r["frustrated"] else 0.0,
            "explanation": f"Human label assigned before evaluation ({r['conversation']}, turn {r['turn']}).",
        })

    client.spans.log_span_annotations_dataframe(
        dataframe=pd.DataFrame(rows), annotation_name=ANNOTATION_NAME, annotator_kind="HUMAN", sync=True
    )
    frustrated = sum(row["score"] for row in rows)
    print(f"Logged {len(rows)} '{ANNOTATION_NAME}' span labels ({frustrated:.0f} frustrated)")

    # Session level: a conversation is frustrated if any of its turns is
    session_rows = []
    for session_id in dict.fromkeys(r["session_id"] for r in results):
        turns = [r for r in results if r["session_id"] == session_id]
        frustrated_turns = [str(r["turn"]) for r in turns if r["frustrated"]]
        session_rows.append({
            "session_id": session_id,
            "label": "frustrated" if frustrated_turns else "not_frustrated",
            "score": 1.0 if frustrated_turns else 0.0,
            "explanation": f"Human label ({turns[0]['conversation']}): frustrated in turn(s) {', '.join(frustrated_turns)}."
            if frustrated_turns else f"Human label ({turns[0]['conversation']}): no frustrated turn.",
        })

    client.sessions.log_session_annotations_dataframe(
        dataframe=pd.DataFrame(session_rows), annotation_name=ANNOTATION_NAME, annotator_kind="HUMAN", sync=True
    )
    frustrated = sum(row["score"] for row in session_rows)
    print(f"Logged {len(session_rows)} '{ANNOTATION_NAME}' session labels ({frustrated:.0f} frustrated)")


if __name__ == "__main__":
    main()
