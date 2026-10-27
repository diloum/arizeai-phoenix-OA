# Travel Agent with Phoenix Tracing and Evaluation

A LangGraph travel assistant served by FastAPI. It can search the web and look up hotels, it remembers the conversation across messages, every request is traced in Arize Phoenix, and a user frustration evaluation runs on those traces.

Built on the starter repo [trevorlaviale/se-interview](https://github.com/trevorlaviale/se-interview).

## How the agent works

The agent is a small LangGraph graph with two nodes:

- `llm_call` sends the conversation to GPT-4o (temperature 0) with the tools attached.
- `tool_node` runs whatever tools the model asked for and adds the results to the conversation.

After `llm_call`, if the model asked for a tool the graph goes to `tool_node` and then back to `llm_call`. If it didn't, the graph ends and the model's message is the answer. This is the usual ReAct loop: think, act, look at the result, repeat.

```
START -> llm_call --(tool calls?)--> tool_node
            ^   \                      |
            |    `--(no)--> END        |
            `--------------------------'
```

Tools:

- `duckduckgo_search`: web search (from the starter repo), for attractions, transport, anything current.
- `get_hotel_options`: hotel lookup that I added, see below.

### Conversation memory

`/chat` takes an optional `session_id` and always returns one. Without it a new conversation starts; sending it back continues the same conversation, so a follow-up like "make that 4 nights" works.

This uses LangGraph's `MemorySaver` checkpointer, with the `session_id` as LangGraph's `thread_id`. It stores the history in the server's memory, so it is lost when the app restarts. That's fine for a demo; in production I'd use a database-backed checkpointer.

## The hotel tool (`hotel_tool.py`)

The brief asked for a tool that fits the workflow, gets called when relevant and returns structured output. Design decisions:

- **Mock data on purpose.** A fixed inventory for Lisbon, Paris, Marseille and Tokyo (prices per night in EUR). Results are repeatable, which makes the demo reliable and the evaluations comparable between runs. In production this function would call a booking API instead.
- **Typed input.** `city`, `check_in` (YYYY-MM-DD), `nights` and an optional `max_price_per_night`, defined with a Pydantic schema so the model gets clear argument descriptions.
- **Structured output.** Always the same JSON shape: `status` (`ok`, `no_results` or `error`), the search parameters, and a list of options sorted by price with the total for the stay.
- **Errors are returned, not raised.** A bad date, zero nights, an unknown city or an empty budget come back as a `status` with a message (an unknown city also lists the cities that are available). The model can then explain the problem to the user instead of the request failing.
- **The docstring does the routing.** The model picks tools from their descriptions, so the docstring says when to use it (where to stay, hotel prices) and not to use web search for hotels.
- **No graph changes.** Adding the tool to the `tools` list in `agent.py` is enough: the model's tool list and the tool runner are both built from it.

## Observability with Phoenix

Tracing is set up in `agent.py` with one call:

```python
register(project_name="travel-agent-arize-OA", auto_instrument=True)
```

`arize-phoenix-otel` sets up OpenTelemetry to send traces to Phoenix, and `openinference-instrumentation-langchain` (which also covers LangGraph) creates the spans. Every request shows up as a trace with the graph, each LLM call and each tool call as spans. I used the Phoenix CLI (`px setup`) and the Phoenix skills for Claude Code to set this up.

OpenTelemetry is the standard way to collect and send traces. OpenInference adds conventions on top for AI apps: span kinds (LLM, TOOL, CHAIN...) and attribute names for prompts, tool calls and token counts.

A **span** is one step (one LLM call, one tool call). A **trace** is one request from start to finish, made of nested spans. A **session** groups the traces of one conversation: `api.py` wraps each request in `using_session(session_id)`, so every span carries the `session.id` and Phoenix shows the conversation in its Sessions tab.

## Setup

You need Python 3.11 to 3.14 (I used 3.13; some dependencies don't install on 3.15 yet), [Poetry](https://python-poetry.org/docs/#installation) and an OpenAI API key. Node.js is only needed for the `px` CLI.

1. Install the dependencies and add your key:

   ```bash
   poetry env use python3.13
   poetry install
   cp .env.example .env    # then set OPENAI_API_KEY in .env
   ```

2. Start Phoenix in its own environment and terminal. Start it before the app, so no traces are lost:

   ```bash
   python3.13 -m venv ~/phoenix-env
   ~/phoenix-env/bin/pip install arize-phoenix
   ~/phoenix-env/bin/phoenix serve
   ```

   The UI is at http://localhost:6006. The app sends traces there by default; set `PHOENIX_COLLECTOR_ENDPOINT` if Phoenix runs somewhere else.

3. Start the app in another terminal:

   ```bash
   poetry run uvicorn api:app --reload
   ```

   The API is at http://localhost:8000 (interactive docs at http://localhost:8000/docs).

## Using the API

Start a conversation:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find me a hotel in Lisbon from 2026-11-10 for 3 nights."}'
```

```json
{"response": "Here are some hotel options in Lisbon...", "session_id": "cb362b79-4cb8-445d-b31b-2e797d3d4709"}
```

Continue it by sending the `session_id` back:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Actually, under 100 euros a night.", "session_id": "cb362b79-4cb8-445d-b31b-2e797d3d4709"}'
```

`GET /health` returns `{"status": "ok"}`.

## Evaluation: user frustration

### Test data

- `send_queries.py`: 16 single-turn queries. Hotel requests, edge cases (unknown city, nothing in budget, vague date), web search, both tools, no tool, 4 frustrated messages and 2 neutral look-alikes. Results in `query_results.json`.
- `send_sessions.py`: 7 multi-turn conversations, 18 turns. Some go well, some make the user frustrated (unknown city, vague date, nothing in budget, ignored budget), and some include corrections that are not complaints ("Actually, make it 3 nights instead of 2"). Results in `session_results.json`.

I labelled every message as frustrated or not before running any evaluation. These labels are the ground truth.

Frustration depends on context. "I mean Friday 2026-11-06. Why do I have to spell everything out?" only makes sense after the previous turn, which is why I added conversation memory and sessions.

### Method

`run_frustration_eval.py` runs Phoenix's built-in `UserFrictionEvaluator`, an LLM-as-a-judge classifier, with gpt-4o-mini at temperature 0. For each turn it gives the judge two inputs, kept separate as the Phoenix docs recommend:

- `conversation`: the earlier user and assistant messages of the session (tool results left out)
- `user_message`: the message sent in this turn

Both come from the turn's root span in Phoenix, whose output holds the whole conversation up to that turn, so the evaluation reads the traces rather than my local files. The judge returns `friction` / `no_friction`, a score (1 for friction, 0 for none) and an explanation.

Results are attached to the traces as `user_frustration` annotations (annotator kind `LLM`):

- on each turn's root span, with `log_span_annotations_dataframe`
- on each session, with `log_session_annotations_dataframe` (a session counts as friction if any of its turns does)

My labels are attached the same way as `human_frustration` (annotator kind `HUMAN`) by `log_human_labels.py`, so the judge's label and mine sit side by side on the same spans.

In Phoenix I then filtered the spans with `annotations['user_frustration'].label == 'friction'` and saved the matches as the dataset `friction_eval_dataset`. `frustrated_human_label_interactions` holds the turns I labelled as frustrated.

### Results

Phoenix holds 62 turns: 44 in 20 sessions and 18 single-turn requests, including earlier test runs and manual tests. The judge flagged 11 turns and 7 sessions as friction, and I saved the 11 flagged turns as `friction_eval_dataset`.

The evaluator only returns `friction` or `no_friction`. Its rubric looks for four kinds of signal (correction, retry, frustration, challenge), but which one it found only appears in the explanation text, not as a separate field. In my flagged turns, most explanations point to frustration ("Ugh", "Why is that so hard?") and a few to a correction or a retry ("That's wrong, I said Marseille").

Two of the flagged turns are mistakes by the judge's own rubric:

- "Actually my budget is 100 euros a night max." The rubric says a new constraint that builds on the previous answer is `no_friction`. It handled the similar "Actually, make it 3 nights instead of 2" correctly.
- "Sorry, I meant Marseille, not Paris." Borderline: it redirects the assistant, but the mistake was the user's, not the assistant's.

The rubric also says to prefer `no_friction` when in doubt, because precision on the friction label matters more than recall. So these are precision errors by the judge model (gpt-4o-mini), not a difference in definitions. Next I'd try a stronger judge model, and a custom prompt that also returns which signal it found, so the type becomes something you can filter on.

Also from the docs: `no_friction` doesn't mean the user was happy. People often just leave.

### Found in the traces

Asked for a hotel "next Friday", the agent called the hotel tool with `check_in: 2023-11-03`, a date in the past. The model doesn't know today's date and the tool accepts past dates. The answer to the user said "from next Friday" and looked fine; only the tool span shows the problem. A likely fix is putting today's date in the system prompt and rejecting past dates in the tool.

### Reproducing it

With Phoenix and the app running:

```bash
poetry run python send_queries.py          # single-turn queries
poetry run python send_sessions.py         # multi-turn conversations
poetry run python log_human_labels.py      # human labels -> Phoenix
poetry run python run_frustration_eval.py  # LLM judge -> Phoenix
```

Export the spans with their annotations (this is how `spans.json` was made):

```bash
px span list spans.json --project travel-agent-arize-OA --limit 5000 --include-annotations
```

## Limitations and next steps

- The users are scripted, so they don't react to what the agent actually says. Phoenix's session evaluation cookbook uses a simulated user for that.
- Few labelled examples, all written by me. To trust the judge on real traffic I'd want more, ideally labelled by someone else.
- Try a custom judge that returns the type of friction (correction, retry, frustration, challenge) and compare it with the built-in one on `friction_eval_dataset`.
- Fix the date issue above, then re-run the evaluation to check it.
- Other evaluators worth adding: tool selection (did it pick the hotel tool for hotel questions?) and tool invocation (were the arguments right?).
- Memory is in-process only; a persistent checkpointer for anything beyond a demo.

## Project structure

```
se-interview/
├── agent.py                  # LangGraph agent, tools, Phoenix tracing, memory
├── api.py                    # FastAPI server: /chat (with session_id) and /health
├── hotel_tool.py             # get_hotel_options tool (mock inventory)
├── send_queries.py           # 16 single-turn test queries
├── send_sessions.py          # 7 labelled multi-turn conversations
├── log_human_labels.py       # human frustration labels -> Phoenix
├── run_frustration_eval.py   # User Friction judge -> Phoenix
├── query_results.json        # answers to send_queries.py
├── session_results.json      # answers to send_sessions.py
├── spans.json                # spans exported from Phoenix, with annotations
├── pyproject.toml / poetry.lock
├── .env.example
└── .agents/, .claude/skills/ # Phoenix skills used with Claude Code
```

## How I built it

I used Claude Code with the Phoenix CLI and skills for the instrumentation, the conversation memory and the evaluation scripts, and Claude to research Phoenix's documentation and plan the tests. I reviewed each change before it went in.
