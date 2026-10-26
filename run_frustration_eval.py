"""Run Phoenix's built-in User Friction evaluator (LLM-as-a-judge) on every /chat turn.

For each turn (the root LangGraph span of a trace), the judge gets:
  - conversation: the earlier user and assistant messages of the session (tool payloads left out)
  - user_message: the message sent in this turn
Both come from the span itself: its output holds the conversation up to and including this turn.

Results are logged to Phoenix as "user_frustration" (annotator_kind="LLM") on each span,
and on each session (frustrated if any turn is). Then the judge is compared with the
human labels logged by log_human_labels.py.

Run from the project folder with Phoenix running (needs OPENAI_API_KEY in .env):
    poetry run python run_frustration_eval.py
"""

import json

import pandas as pd
from dotenv import load_dotenv
from phoenix.client import Client
from phoenix.evals import LLM
from phoenix.evals.metrics import UserFrictionEvaluator

load_dotenv()

PROJECT_NAME = "travel-agent-arize-OA"
ANNOTATION_NAME = "user_frustration"
HUMAN_ANNOTATION_NAME = "human_frustration"


def split_turn(output_value: str) -> tuple[str, str]:
    """Return (conversation before this turn's user message, this turn's user message)."""
    messages = json.loads(output_value)["messages"]
    last_user = max(i for i, m in enumerate(messages) if m["type"] == "human")
    lines = [
        f"{'User' if m['type'] == 'human' else 'Assistant'}: {m['data']['content']}"
        for m in messages[:last_user]
        if m["type"] in ("human", "ai") and m["data"]["content"]  # skip tool results and tool-call-only steps
    ]
    return "\n".join(lines) or "(no earlier messages)", messages[last_user]["data"]["content"]


def main() -> None:
    client = Client()
    evaluator = UserFrictionEvaluator(llm=LLM(provider="openai", model="gpt-4o-mini"), temperature=0.0)

    spans = client.spans.get_spans_dataframe(project_name=PROJECT_NAME, root_spans_only=True, limit=5000)
    spans = spans[spans["attributes.output.value"].notna()]

    rows = []
    for span_id, span in spans.iterrows():
        conversation, user_message = split_turn(span["attributes.output.value"])
        score = evaluator.evaluate({"conversation": conversation, "user_message": user_message})[0]
        rows.append({
            "span_id": span_id,
            "session_id": span["attributes.session.id"],
            "user_message": user_message,
            "label": score.label,
            "score": score.score,
            "explanation": score.explanation,
        })
        print(f"{score.label:>12}  {user_message[:90]}")
    results = pd.DataFrame(rows)

    client.spans.log_span_annotations_dataframe(
        dataframe=results[["span_id", "label", "score", "explanation"]],
        annotation_name=ANNOTATION_NAME, annotator_kind="LLM", sync=True,
    )
    print(f"\nLogged {len(results)} '{ANNOTATION_NAME}' span annotations "
          f"({(results['label'] == 'friction').sum()} friction)")

    # Session level: a conversation is frustrated if any of its turns is
    session_rows = []
    for session_id, turns in results.dropna(subset=["session_id"]).groupby("session_id"):
        flagged = turns[turns["label"] == "friction"]
        session_rows.append({
            "session_id": session_id,
            "label": "friction" if len(flagged) else "no_friction",
            "score": 1.0 if len(flagged) else 0.0,
            "explanation": " | ".join(f"{m!r}: {e}" for m, e in zip(flagged["user_message"], flagged["explanation"]))
            or "No turn flagged by the judge.",
        })
    client.sessions.log_session_annotations_dataframe(
        dataframe=pd.DataFrame(session_rows), annotation_name=ANNOTATION_NAME, annotator_kind="LLM", sync=True,
    )
    print(f"Logged {len(session_rows)} '{ANNOTATION_NAME}' session annotations "
          f"({sum(r['score'] for r in session_rows):.0f} friction)")

    # Compare the judge with the human labels, on the turns that have both
    human = client.spans.get_span_annotations_dataframe(
        span_ids=list(results["span_id"]), project_identifier=PROJECT_NAME
    )
    human = human[human["annotation_name"] == HUMAN_ANNOTATION_NAME]["result.score"].rename("human_score")
    compared = results.set_index("span_id").join(human, how="inner")
    agree = compared["score"] == compared["human_score"]
    print(f"\nJudge vs human, {len(compared)} labeled turns: {agree.sum()}/{len(compared)} agree ({agree.mean():.0%})")
    for _, row in compared[~agree].iterrows():
        human_label = "frustrated" if row["human_score"] else "not frustrated"
        print(f"  Disagree: human={human_label}, judge={row['label']}: {row['user_message']!r}\n"
              f"    Judge's reason: {row['explanation']}")


if __name__ == "__main__":
    main()
