# Vera — Merchant Engagement AI

Vera is an AI-driven merchant engagement assistant built for the magicpin AI Challenge.

The system receives merchant, customer, category, and trigger context through the challenge API. It decides whether an engagement should be sent, selects the relevant signal, composes a concise WhatsApp-style message, and maintains state across interactions.

## Architecture

```text
                         magicpin Judge
                              │
             ┌────────────────┼────────────────┐
             │                │                │
          /context           /tick           /reply
             │                │                │
             ▼                ▼                ▼
      ┌────────────────────────────────────────────┐
      │                  FastAPI                   │
      │                   bot.py                   │
      │                                            │
      │  • context ingestion                       │
      │  • versioned context storage               │
      │  • conversation state                      │
      │  • suppression state                       │
      │  • tick orchestration                      │
      └──────────────────────┬─────────────────────┘
                             │
                             ▼
      ┌────────────────────────────────────────────┐
      │                  engine.py                 │
      │                                            │
      │  • send decision / gating                  │
      │  • compact prompt construction              │
      │  • LLM composition                         │
      │  • JSON extraction / validation             │
      │  • repetition detection                    │
      │  • deterministic fallback                  │
      └──────────────────────┬─────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Gemini Flash   │
                    │ temperature = 0 │
                    └────────┬────────┘
                             │
                             ▼
                    Engagement Action
                 body / CTA / send_as /
                       rationale
```

## Request Flow

### 1. Context ingestion

`POST /v1/context` receives context from the judge and stores it using:

- scope
- context ID
- version
- payload

Newer versions replace older versions, while stale versions are rejected.

The application applies a 450 KB safety threshold to the serialized context payload, leaving headroom below the challenge's 500 KB request limit.

### 2. Tick processing

`POST /v1/tick` receives the currently available triggers.

For each trigger, Vera resolves:

- trigger
- merchant
- category
- optional customer
- relevant conversation state

A deterministic send gate first checks whether there is a meaningful reason to engage.

Only eligible candidates proceed to message composition.

Independent compositions are executed concurrently so that multiple LLM calls can complete within the challenge's 30-second per-call limit.

The number of returned actions is capped at 20.

### 3. Message composition

`engine.py` builds a compact prompt containing only the relevant context.

The composer prioritizes:

1. Continuing an active conversation when appropriate.
2. Using the most relevant current signal.
3. Personalizing from actual merchant/customer/category information.
4. Providing one clear reason to respond.
5. Asking for one primary next action.

The LLM is required to return structured JSON containing:

```text
body
cta
send_as
rationale
```

### 4. Validation and safeguards

The generated response is validated before being returned as an action.

Vera also applies a repetition guard against recently generated messages.

If the LLM:

- times out,
- fails,
- returns malformed JSON,
- returns an empty message, or
- otherwise fails composition,

Vera falls back to deterministic message generation.

This prevents an LLM failure from causing the complete request to fail.

### 5. Reply handling

`POST /v1/reply` records merchant/customer replies and updates conversation state.

Subsequent decisions can therefore prioritize the current conversation instead of treating every trigger as an independent interaction.

## Message Composition Principles

Vera's composer is designed around the engagement requirements of the challenge.

### Grounded messaging

Messages use information received from the judge rather than inventing facts.

The composer must not invent:

- prices
- dates
- percentages
- offers
- availability
- statistics
- performance values
- other unsupported claims

### Specificity

When a concrete signal exists, Vera uses that signal instead of producing a generic follow-up.

Examples of useful signals include:

- current metrics
- changes or deltas
- deadlines
- offers
- festivals or relevant events
- competitor information
- customer intent
- merchant performance information

### Conversation awareness

If recent merchant/customer conversation exists, it is prioritized over starting a disconnected generic message.

### Personalization

The message is adapted to the available:

- merchant information
- customer information
- category
- trigger
- current conversation

When customer context is present, the action can be sent on behalf of the merchant.

Otherwise, Vera speaks as the assistant.

### Controlled calls to action

Each message should have one clear reason to respond and one primary next action.

### Repetition control

Recent messages are compared against a normalized representation of the new message to reduce repetitive outreach.

## Decision Layer

The decision layer intentionally remains separate from the LLM composer.

The send decision is based on received evidence such as:

- active merchant message
- active customer message
- customer intent
- current headline/signal
- metric
- percentage change
- offer
- deadline
- festival/theme
- competitor information
- other meaningful trigger fields

This prevents the LLM from being solely responsible for deciding whether every trigger deserves an engagement.

## State

Vera maintains in-memory state for the running service:

```text
contexts
    └── (scope, context_id)
            ├── version
            └── payload

conversations
    └── conversation_id
            ├── merchant/customer messages
            └── latest state

suppression
    └── previously sent suppression keys

merchant_state
    └── merchant-level interaction state
```

The state is designed for the challenge's stateful evaluation flow.

The bot does not require the challenge seed dataset at runtime. Merchant, customer, category, and trigger information is received through the API.

## API Contract

```text
GET  /v1/healthz
GET  /v1/metadata
POST /v1/context
POST /v1/tick
POST /v1/reply
```

### `GET /v1/healthz`

Health endpoint used by the evaluation harness to verify that the service is alive.

### `GET /v1/metadata`

Returns bot/team metadata required by the challenge interface.

### `POST /v1/context`

Receives versioned context from the judge.

The endpoint is designed to be idempotent with respect to scope, context ID, and version.

### `POST /v1/tick`

Evaluates available triggers and returns engagement actions.

The implementation limits processing to the challenge's maximum action count and composes eligible messages concurrently.

### `POST /v1/reply`

Receives simulated merchant/customer replies and updates conversation state.

## Runtime Constraints

The implementation is designed around the challenge testing constraints.

| Constraint | Implementation |
|---|---|
| `/v1/context` payload | Maximum 500 KB |
| Application context threshold | 450 KB safety limit |
| `/v1/tick` action count | Maximum 20 actions |
| Judge request rate | Maximum 10 requests/second |
| Per-call timeout | 30 seconds |
| LLM timeout | Configurable through `LLM_TIMEOUT` |
| Multiple compositions | Concurrent execution during a tick |
| LLM failure | Deterministic fallback |
| Invalid LLM output | JSON validation + fallback |
| Repeated messages | Similarity/repetition guard |

The 450 KB application threshold intentionally leaves approximately 50 KB of headroom below the challenge's 500 KB request limit.

The LLM timeout is intentionally shorter than the judge timeout so that a slow external model does not consume the entire request budget.

## Reliability Strategy

The runtime is built with several failure boundaries:

```text
HTTP request
    │
    ▼
Context / trigger validation
    │
    ▼
Deterministic send gate
    │
    ▼
Bounded LLM call
    │
    ├── success ──────► JSON validation
    │                       │
    │                       ▼
    │                  repetition guard
    │                       │
    │                       ▼
    │                    action
    │
    └── failure ──────► deterministic fallback
```

The goal is graceful degradation rather than allowing a single model failure to make the entire tick fail.

## Configuration

The deployed service uses environment variables for model configuration and secrets.

```text
LLM_PROVIDER=gemini
GEMINI_API_KEY=<your-api-key>
LLM_MODEL=gemini-3.8-flash
LLM_TIMEOUT=6
```

API keys and other secrets are never stored in source control.

## Local Development

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI service:

```bash
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Local base URL:

```text
http://localhost:8080
```

The supplied local judge simulator and API examples can be used to validate:

- endpoint availability
- payload handling
- response shape
- state handling
- reply handling
- timeout behavior
- canonical test scenarios

## Deployment

The application is packaged as a FastAPI service and can be deployed as a public HTTPS endpoint.

The production service must expose the same base API contract:

```text
https://<public-host>/v1/healthz
https://<public-host>/v1/metadata
https://<public-host>/v1/context
https://<public-host>/v1/tick
https://<public-host>/v1/reply
```

The public base URL is submitted to the challenge evaluation harness.

The challenge dataset and local judge simulator are development/testing resources. They are not required by the bot at runtime because the evaluation harness supplies the relevant context through `/v1/context`.

## Project Structure

```text
.
├── bot.py
├── engine.py
├── requirements.txt
├── README.md
├── .gitignore
```

## Design Summary

Vera separates four responsibilities:

```text
Context
   │
   ▼
Decision
   │
   ▼
Composition
   │
   ▼
Validation / Safeguards
   │
   ▼
Action
```

The LLM is therefore used primarily for natural-language composition, while deterministic application logic controls context handling, engagement gating, state, validation, repetition protection, timeout behavior, and fallback behavior.

This separation allows the system to remain grounded in received context while satisfying the challenge's latency, payload, action-count, and reliability constraints.
