# Hypercare Agents

[![CI](https://github.com/zeineeb33/hypercare-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/zeineeb33/hypercare-agents/actions/workflows/ci.yml)

A multi-agent system that handles support tickets during the **hypercare** phase of a software rollout: the weeks after go-live, when ticket volume spikes and the documentation is still catching up.

It answers from the documentation and the data, never invents a price, lets an independent verification agent veto unsafe answers, and detects recurring questions the documentation cannot answer, then drafts and tests a fix for a human to approve.

The domain is the pricing platform of a container shipping carrier. **The company (Northwind Shipping), its products (TideRate, QuotePort), the data and the documents are fictional.**

> Live demo: *coming soon* (Render, replay mode).

## Architecture

```mermaid
flowchart TD
    T[Ticket] --> O[Orchestrator<br/>classification]
    O -->|pricing ticket without any reference| C[ASK_CLARIFICATION]
    O --> K[Knowledge agent<br/>query planning, hybrid retrieval,<br/>grounded answer with citations]
    O --> D[Data investigation agent<br/>parameterised SQL: quote, price line,<br/>surcharges, expiry, overlaps, inland rates]
    K --> P{Recurring and<br/>undocumented?}
    D --> P
    P -->|yes| G[DETECT_KB_GAP]
    P -->|no| R[Resolution agent<br/>structured decision + reply]
    R --> V[Verification agent<br/>6 deterministic checks, can veto]
    G --> V
    C --> V
    V --> F[ANSWER / ESCALATE /<br/>ASK_CLARIFICATION / DETECT_KB_GAP]

    M[Maintenance agent] -->|recurring gap| PR[Drafts a new section]
    PR --> TST[Pre-publication tests:<br/>does the updated doc answer the question?]
    TST --> H[Human review] -->|approve| KB[(Versioned knowledge base)]
```

### The agents

| Agent | Type | Role |
|---|---|---|
| Orchestrator | Code | Routes the ticket, asks for clarification when a pricing ticket has nothing to investigate, traces every step |
| Knowledge agent | Claude (2 calls) | Writes search queries, retrieves sections (BM25, optionally fused with embeddings), answers **only** from them, cites them, flags insufficient evidence |
| Data investigation agent | SQL | Follows the references in the ticket (`QS-…`, `BK-…`, `PL-… line …`) and looks for the usual root causes |
| Pattern detection | Code | Many open tickets in a subcategory **and** a documentation that cannot answer them = knowledge gap |
| Resolution agent | Claude | Decides `ANSWER` / `ASK_CLARIFICATION` / `ESCALATE` and writes the reply, with the evidence used |
| Verification agent | Code | Six checks on the proposed reply; a blocking failure turns it into `ESCALATE`. **Fails closed**: if it cannot run, the ticket is escalated |
| Maintenance agent | Claude + code | Drafts a documentation update for each gap, tests it, stores it for review; approval versions and publishes the document |

### Verification checks

| Check | Blocks |
|---|---|
| Answered although the documentation is insufficient and no data supports the answer | yes |
| Answer given on a `DRAFT` (unpublished) price line | yes |
| Expired rate presented as current | yes |
| No source, or a cited document that was never retrieved | yes |
| Overlapping `ACTIVE` lines (hypercare must not choose) | yes |
| Pricing answer that states none of the actual amounts | warning |

## Engineering decisions

- **Structured outputs everywhere.** Every Claude call goes through `messages.parse` with a Pydantic schema, so agents exchange validated objects, not free text. Any API error, refusal, truncated or invalid output becomes an `LLMError`, and the pipeline escalates.
- **Deterministic where it matters.** SQL investigation, pattern detection and verification are plain code: they are testable, cheap and cannot hallucinate. The LLM writes; code decides whether the answer is safe to send.
- **Fail closed.** No path lets an unverified answer through: a crash in verification, a failed LLM call or a missing replay all end in an escalation.
- **Pre-publication tests that test something.** A documentation proposal is accepted only if the knowledge agent, re-run on the original question against the updated document, can now answer it. The model is told to leave `[placeholders]` instead of inventing owners or delays; a proposal with placeholders cannot be approved until the reviewer fills them, and the tests run again on the edited text.
- **Replay mode.** Only the LLM calls are recorded and replayed (keyed by a hash of the prompt). Retrieval, SQL and verification still run live in the public demo, at zero cost.
- **Small footprint.** BM25 is pure Python, the database is SQLite, so the demo fits Render's free tier. Dense embeddings (`fastembed`) are an optional extra.
- **Claude Opus 5.5** with adaptive thinking at `medium` effort, cacheable system prompts and server-side refusal fallbacks. The model is one setting (`HYPERCARE_MODEL`).

## Quickstart

```bash
uv sync                                   # Python 3.12
uv run hypercare seed                     # build the SQLite database
uv run hypercare serve                    # Streamlit app on http://localhost:8501 (replay mode)
uv run pytest --cov                       # tests: no network, no API key
```

Live mode (calls Claude):

```bash
cp .env.example .env                      # then put your ANTHROPIC_API_KEY in .env
uv run hypercare run 3                    # one ticket, with its trace and checks
uv run hypercare eval                     # the 27 evaluation cases
uv run hypercare maintain                 # detect gaps and draft documentation updates
uv run hypercare review 1 --reviewer you --approve
```

`--record` on `run`, `eval` and `maintain` saves the LLM answers for replay.

## Results

27 evaluation cases covering the six scenarios plus documentation, booking, discount, incident and expiry questions ([`cases.json`](src/hypercare/data/eval/cases.json)):

| Metric | Result |
|---|---|
| Decision accuracy (`ANSWER` / `ESCALATE` / `ASK_CLARIFICATION` / `DETECT_KB_GAP`) | 27 / 27 |
| Answers citing the expected document | 100% |
| Answers stating the expected facts (amounts, contacts, delays) | 100% |
| Self-healing loop | gap detected on the inland-rate tickets only, proposal blocked until the reviewer fills its placeholders, approved, and the same five tickets then get an `ANSWER` |

**How to read this.** The LLM answers used here were produced by Claude Opus 5.5 through Claude Code (see [Recording without the API](#recording-without-the-api)), by the same session that wrote the cases, so these numbers show that the pipeline works end to end; they are not an independent benchmark. Latency and cost are not measured for replayed answers. `HYPERCARE_MODE=live uv run hypercare eval` runs the same cases against the API for an independent measurement.

The verification agent overrode nothing in this run because every proposed reply was safe. Its vetoes are covered by the tests, and the app has a **fault injection** switch that forces the resolution agent to answer so you can watch the checks block it (try ticket HC-2026-0006, a DRAFT line).

## Recording without the API

The public demo replays recorded LLM answers. They can come from a live run (`--record`) or be produced outside the app, for example in Claude Code, without API billing:

```bash
uv run hypercare prompts export           # run every ticket, write the prompts that have no answer yet
uv run hypercare prompts show             # read them
uv run hypercare prompts import answers.json   # validate each answer against its schema and store it
```

Repeat until nothing is pending: each agent waits for the previous one's answer (plan, then retrieval and answer, then decision). The export replays the whole demo, including the maintenance agent, the reviewer filling the placeholders ([`reviewer_edits.json`](src/hypercare/data/demo/reviewer_edits.json)), the approval and the re-run of the gap tickets.

## Project structure

```
src/hypercare/
  agents/         orchestrator, knowledge, data investigation, resolution, verification, maintenance
  llm/            Anthropic client (structured outputs), replay/recording clients, fake for tests
  retrieval/      section chunking, BM25, optional embeddings, reciprocal rank fusion
  data/seed/      synthetic pricing data and 30 tickets
  data/kb/        fictional knowledge base (11 documents, including a video transcript)
  data/eval/      evaluation cases
  ui/             Streamlit app
  schema.sql      SQLite schema
tests/            pytest (fake LLM, local stub server for the SDK, Streamlit AppTest)
snowflake/        first prototype on Snowflake Cortex (see below)
```

## Background

The first prototype ran entirely inside Snowflake: Snowpark stored procedures, Cortex Search, Cortex `COMPLETE`, Streams and Tasks, and Streamlit in Snowflake. It is kept in [`snowflake/`](snowflake/). This version keeps its architecture and scenarios, and changes what a production version needs: portable Python, validated structured outputs, parameterised SQL, a verification step that fails closed, real pre-publication tests, an evaluation set and a test suite.

## License

MIT
