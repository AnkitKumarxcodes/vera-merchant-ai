"""
engine.py — Vera's message decision and composition engine.

Contains the LLM interface and the logic used to decide whether a
trigger should produce a message and how that message should be
composed.

This module is independent of the FastAPI endpoint layer.
"""


import json
import os
import re
import time
from typing import Optional
from urllib import request as urlreq

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "6"))


def llm_complete(system: str, user: str) -> str:
    if LLM_PROVIDER == "ollama":
        model = LLM_MODEL or "llama3"
        body = json.dumps({
            "model": model, "prompt": f"{system}\n\n{user}", "stream": False,
            "options": {"temperature": 0, "num_predict": 200},
        }).encode()
        req = urlreq.Request(f"{OLLAMA_URL}/api/generate", data=body,
                              headers={"content-type": "application/json"})
        with urlreq.urlopen(req, timeout=LLM_TIMEOUT) as r:
            return json.loads(r.read())["response"]
    elif LLM_PROVIDER == "anthropic":
        model = LLM_MODEL or "claude-sonnet-4-6"
        body = json.dumps({
            "model": model, "max_tokens": 500, "temperature": 0,
            "system": system, "messages": [{"role": "user", "content": user}],
        }).encode()
        req = urlreq.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"x-api-key": LLM_API_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        with urlreq.urlopen(req, timeout=LLM_TIMEOUT) as r:
            return json.loads(r.read())["content"][0]["text"]
    elif LLM_PROVIDER == "openai":
        model = LLM_MODEL or "gpt-4o-mini"
        body = json.dumps({
            "model": model, "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }).encode()
        req = urlreq.Request(
            "https://api.openai.com/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {LLM_API_KEY}", "content-type": "application/json"})
        with urlreq.urlopen(req, timeout=LLM_TIMEOUT) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    elif LLM_PROVIDER == "gemini":
        model = LLM_MODEL or "gemini-3.8-flash"

        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0
        }).encode()

        req = urlreq.Request(
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "content-type": "application/json"
            }
        )

        start = time.time()

        with urlreq.urlopen(req, timeout=LLM_TIMEOUT) as r:
            result = json.loads(r.read())["choices"][0]["message"]["content"]

        print(f"[LLM] Gemini completed in {time.time() - start:.2f}s")

        return result
        
    raise ValueError(f"unknown LLM_PROVIDER: {LLM_PROVIDER}")

def _extract_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    return json.loads(match.group()) if match else {}

# --------------------------------------------------------------------------
# Composer — the 4-context → message engine
# --------------------------------------------------------------------------

COMPOSER_SYSTEM = """
You are Vera, magicpin's merchant-growth WhatsApp assistant.

Compose ONE short, natural WhatsApp message using the supplied context.

PRIORITY:
1. Continue the current conversation if a recent merchant/customer message exists.
2. Otherwise use the most relevant fact from the current trigger.
3. Personalize using the actual merchant/customer information.
4. Give one clear reason to respond.
5. Ask for one next action only.

STRICT RULES:
- Never invent prices, dates, percentages, offers, availability or statistics.
- Never mention internal fields, IDs, payloads, trigger names or dataset terminology.
- Never produce a generic "following up" message when a concrete fact is available.
- Do not repeat an old message from avoid_messages.
- Keep the message concise and WhatsApp-natural.
- Match the category's voice.
- If customer context exists, write to the CUSTOMER and use send_as="merchant_on_behalf".
- Otherwise write to the MERCHANT and use send_as="vera".

Return ONLY valid JSON:

{
  "body": "...",
  "cta": "binary|open_ended|none",
  "send_as": "vera|merchant_on_behalf",
  "rationale": "..."
}
"""

def build_user_prompt(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict],
    avoid_bodies: list[str],
) -> str:

    merchant_identity = merchant.get("identity", {})
    payload = trigger.get("payload", {}) or {}

    context = {
        "category": {
            "name": category.get("display_name") or category.get("name"),
            "slug": category.get("slug"),
            "voice": category.get("voice"),
            "taboo_words": category.get("taboo_words"),
        },

        "merchant": {
            "name": merchant_identity.get("name"),
            "city": merchant_identity.get("city"),
            "language": merchant_identity.get("language_preference"),
            "offers": merchant.get("offers"),
            "signals": merchant.get("signals"),
            "metrics": merchant.get("metrics"),
        },

        "trigger": {
            "kind": trigger.get("kind"),
            "payload": payload,
        },
    }

    if customer:
        context["customer"] = {
            "name": customer.get("identity", {}).get("name"),
            "language": customer.get("identity", {}).get("language_preference"),
            "history": customer.get("history"),
            "preferences": customer.get("preferences"),
        }

    if avoid_bodies:
        context["avoid_messages"] = avoid_bodies[-3:]

    return (
        "Use the following context to compose the message. "
        "Only use facts present here; never invent values.\n\n"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    )

def is_too_similar(body: str, previous: list[str]) -> bool:
    if not previous:
        return False

    def normalize(text: str) -> set[str]:
        words = re.findall(r"[a-z0-9₹]+", text.lower())
        return {w for w in words if len(w) > 3}

    current = normalize(body)

    if not current:
        return True

    return any(
        len(current & normalize(old)) / len(current) > 0.65
        for old in previous[-5:]
        if old
    )

def deterministic_fallback(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
) -> dict:

    name = merchant.get("identity", {}).get("name", "there")
    payload = trigger.get("payload", {}) or {}
    kind = trigger.get("kind", "update").replace("_", " ")

    # Prefer actual conversational context.
    conversation = (
        payload.get("merchant_last_message")
        or payload.get("customer_last_message")
        or payload.get("last_message")
    )

    # Otherwise extract the most useful human-readable signal.
    signal = next(
        (
            str(payload[key]).replace("_", " ")
            for key in (
                "headline",
                "intent_topic",
                "metric_or_topic",
                "theme",
                "festival",
                "offer",
                "signal",
                "metric",
                "molecule",
                "competitor_name",
            )
            if payload.get(key)
        ),
        None,
    )

    if conversation:
        body = f"{name} — picking up from your message: “{conversation}” What would you like to do next?"
    elif signal:
        body = f"{name} — {signal}. Want to look at the next step?"
    else:
        body = f"{name} — there’s a {kind} update relevant to your business. Want to take the next step?"

    return {
        "body": body,
        "cta": "open_ended",
        "send_as": "merchant_on_behalf" if customer else "vera",
        "rationale": "deterministic grounded fallback",
    }

def should_send(trigger: dict) -> bool:
    payload = trigger.get("payload", {}) or {}

    # Explicit conversation or intent is inherently actionable.
    if any(
        payload.get(key)
        for key in (
            "merchant_last_message",
            "customer_last_message",
            "intent_topic",
        )
    ):
        return True

    # Don't send if the trigger is only a placeholder with no useful signal.
    meaningful_fields = (
        "headline",
        "signal",
        "metric",
        "delta_pct",
        "offer",
        "deadline",
        "deadline_iso",
        "due_date",
        "days_remaining",
        "days_until",
        "festival",
        "theme",
        "competitor_name",
        "molecule",
        "trends",
        "credits",
        "value_now",
    )

    return any(
        payload.get(field) not in (None, "", [], {})
        for field in meaningful_fields
    )

def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    avoid_bodies: Optional[list[str]] = None,
) -> dict:

    user_prompt = build_user_prompt(
        category,
        merchant,
        trigger,
        customer,
        avoid_bodies or [],
    )

    try:
        raw = llm_complete(COMPOSER_SYSTEM, user_prompt)
        data = _extract_json(raw)

        body = (data.get("body") or "").strip()

        if not body:
            raise ValueError("empty body from LLM")

        if is_too_similar(body, avoid_bodies or []):
            raise ValueError("generated message too similar to previous message")

        return {
            "body": body,
            "cta": data.get("cta", "open_ended"),
            "send_as": data.get(
                "send_as",
                "merchant_on_behalf" if customer else "vera",
            ),
            "rationale": data.get(
                "rationale",
                "composed from supplied context",
            ),
        }

    except Exception as e:
        fallback = deterministic_fallback(
            category,
            merchant,
            trigger,
            customer,
        )

        fallback["rationale"] = (
            f"deterministic fallback after composer failure: "
            f"{type(e).__name__}"
        )

        return fallback