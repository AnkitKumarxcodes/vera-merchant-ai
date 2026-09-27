"""
bot.py — FastAPI endpoint layer for Vera.
"""
import os
import time
import uuid
import re
import json
from collections import Counter
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from engine import (
    LLM_MODEL,
    COMPOSER_SYSTEM,
    build_user_prompt,
    compose_message,
    should_send,
    _extract_json,
    llm_complete_with_retry,
)
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "")
# --------------------------------------------------------------------------
# In-memory state (persists for the life of the process, wiped on /v1/teardown)
# --------------------------------------------------------------------------

contexts: dict[tuple[str, str], dict] = {}          # (scope, id) -> {version, payload}
conversations: dict[str, dict] = {}                 # conv_id -> {merchant_id, customer_id, history[], sent_bodies[]}
sent_suppression_keys: set[str] = set()             # trigger dedup
merchant_state: dict[str, dict] = {}                # merchant_id -> {msg_counts, autoreply_strikes, ended}
START = time.time()


def get_ctx(scope: str, cid: str) -> Optional[dict]:
    entry = contexts.get((scope, cid))
    return entry["payload"] if entry else None



# --------------------------------------------------------------------------
# Conversation-turn heuristics (auto-reply / intent / hostile / decline)
# --------------------------------------------------------------------------

AUTOREPLY_PATTERNS = re.compile(
    r"thank you for contacting|will (respond|get back|reply) shortly|our team will|"
    r"automated (assistant|reply|response)|currently unavailable|out of office|"
    r"shukriya|hamari team tak pahuncha", re.IGNORECASE)

HOSTILE_PATTERNS = re.compile(
    r"stop messaging|this is spam|leave me alone|harassment|useless spam", re.IGNORECASE)

DECLINE_PATTERNS = re.compile(
    r"\bnot interested\b|no thanks|nahi chahiye|don'?t contact|unsubscribe", re.IGNORECASE)

INTENT_PATTERNS = re.compile(
    r"let'?s do it|go ahead|sign me up|i want to join|ok(ay)? lets? do|proceed|sounds good,? do it",
    re.IGNORECASE)


def handle_reply(conv_id: str, merchant_id: str, message: str) -> dict:
    conv = conversations.setdefault(
        conv_id, {"merchant_id": merchant_id, "history": [], "sent_bodies": [], "ended": False})
    conv["history"].append({"from": "merchant", "body": message})

    mstate = merchant_state.setdefault(merchant_id, {"msg_counts": Counter(), "autoreply_strikes": 0})
    mstate["msg_counts"][message] += 1

    if conv["ended"]:
        return {"action": "end", "rationale": "this conversation is already closed"}

    if HOSTILE_PATTERNS.search(message):
        conv["ended"] = True
        return {"action": "end", "rationale": "merchant reacted hostilely; exiting immediately"}

    is_autoreply = bool(AUTOREPLY_PATTERNS.search(message)) or mstate["msg_counts"][message] >= 3
    if is_autoreply:
        if mstate["autoreply_strikes"] == 0:
            mstate["autoreply_strikes"] = 1
            body = "Samajh gayi. Team tak pahunchane se pehle, 2 minute mein khud dekhna chahenge? Chalega?"
            conv["sent_bodies"].append(body)
            return {"action": "send", "body": body, "cta": "binary",
                    "rationale": "first auto-reply detected; one bypass attempt before exiting"}
        conv["ended"] = True
        return {"action": "end", "rationale": "repeated auto-reply pattern; owner unreachable via bot, exiting gracefully"}

    if DECLINE_PATTERNS.search(message):
        conv["ended"] = True
        return {"action": "end", "rationale": "merchant signalled not interested; gracefully exiting conversation"}

    if INTENT_PATTERNS.search(message):
        merchant = get_ctx("merchant", merchant_id) or {}
        merchant_name = merchant.get("identity", {}).get("name", "your business")

        body = (
            f"Perfect, {merchant_name} — let's do it. "
            "I'll take you through the next step and confirm once it's ready."
        )

        conv["sent_bodies"].append(body)
        conv["history"].append({"from": "vera", "body": body})

        return {
            "action": "send",
            "body": body,
            "cta": "none",
            "rationale": "explicit affirmative intent detected; handled deterministically"
        }
    # default: continue the conversation naturally
    merchant = get_ctx("merchant", merchant_id) or {}
    category = get_ctx("category", merchant.get("category_slug", "")) or {}
    try:
        raw = llm_complete_with_retry(
            COMPOSER_SYSTEM,
            build_user_prompt(category, merchant, {"kind": "conversation_reply", "payload": {"message": message}},
                               None, conv["sent_bodies"]))
        data = _extract_json(raw)
        body = (data.get("body") or "").strip() or "Got it — let me know how you'd like to proceed."
        cta = data.get("cta", "open_ended")
    except Exception:
        body, cta = "Got it — let me know how you'd like to proceed.", "open_ended"

    conv["sent_bodies"].append(body)
    conv["history"].append({"from": "vera", "body": body})
    return {"action": "send", "body": body, "cta": cta, "rationale": "continuing conversation from merchant reply"}


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI()


@app.get("/v1/healthz")
def healthz():
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _cid) in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": counts}


@app.get("/v1/metadata")
def metadata():
    return {
        "team_name": os.environ.get("TEAM_NAME", "Team Ankit"),
        "team_members": os.environ.get("TEAM_MEMBERS", "Ankit Kumar").split(","),
        "model": LLM_MODEL or ("claude-sonnet-4-6" if LLM_PROVIDER == "anthropic" else "gpt-4o-mini"),
        "approach": "single-prompt composer over the 4-context input, with rule-based handling for "
                    "auto-reply / intent / hostile / decline turns",
        "contact_email": os.environ.get("CONTACT_EMAIL", ""),
        "version": "0.1.0",
        "submitted_at": datetime.utcnow().isoformat() + "Z",
    }


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str


from fastapi import HTTPException

MAX_CONTEXT_BYTES = 500 * 1024


@app.post("/v1/context")
def context(body: CtxBody):
    payload_size = len(
        json.dumps(body.model_dump(), separators=(",", ":")).encode("utf-8")
    )

    if payload_size > MAX_CONTEXT_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Context payload exceeds 500 KB limit"
        )

    key = (body.scope, body.context_id)
    cur = contexts.get(key)

    if cur and cur["version"] >= body.version:
        return {
            "accepted": False,
            "reason": "stale_version",
            "current_version": cur["version"],
        }

    contexts[key] = {
        "version": body.version,
        "payload": body.payload,
    }

    return {
        "accepted": True,
        "ack_id": f"ack_{body.context_id}_v{body.version}",
        "stored_at": datetime.utcnow().isoformat() + "Z",
    }


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


@app.post("/v1/tick")
def tick(body: TickBody):
    actions = []
    candidates = []

    for trg_id in body.available_triggers:
        if len(candidates) >= 20:
            break

        trigger = get_ctx("trigger", trg_id)

        if not trigger:
            continue

        if trigger.get("suppression_key") in sent_suppression_keys:
            continue

        merchant_id = (
            trigger.get("merchant_id")
            or trigger.get("payload", {}).get("merchant_id")
        )

        merchant = get_ctx("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue

        category = get_ctx(
            "category",
            merchant.get("category_slug", "")
        )

        if not category:
            continue

        customer_id = trigger.get("customer_id")
        customer = (
            get_ctx("customer", customer_id)
            if customer_id
            else None
        )

        if not should_send(trigger):
            continue

        candidates.append(
            (trg_id, trigger, merchant_id, merchant, customer_id, customer)
        )

    def compose_candidate(item):
        trg_id, trigger, merchant_id, merchant, customer_id, customer = item

        composed = compose_message(
            category=get_ctx(
                "category",
                merchant.get("category_slug", "")
            ),
            merchant=merchant,
            trigger=trigger,
            customer=customer,
        )

        return (
            trg_id,
            trigger,
            merchant_id,
            merchant,
            customer_id,
            composed,
        )

    # Gemini free-tier quotas trip on bursts even under the per-minute cap;
    # engine._throttle() already serializes actual calls with a minimum gap,
    # so a small worker count here just keeps requests queued cleanly.
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(compose_candidate, candidates)
        )

    for (
        trg_id,
        trigger,
        merchant_id,
        merchant,
        customer_id,
        composed,
    ) in results:

        conv_id = (
            f"conv_{merchant_id}_{trg_id}_"
            f"{uuid.uuid4().hex[:6]}"
        )

        conversations[conv_id] = {
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "history": [
                {"from": "vera", "body": composed["body"]}
            ],
            "sent_bodies": [composed["body"]],
        }

        sk = trigger.get("suppression_key", "")

        if sk:
            sent_suppression_keys.add(sk)

        actions.append({
            "conversation_id": conv_id,
            "merchant_id": merchant_id,
            "customer_id": customer_id,
            "send_as": composed["send_as"],
            "trigger_id": trg_id,
            "template_name": (
                f"vera_{trigger.get('kind', 'generic')}_v1"
            ),
            "template_params": [
                merchant.get("identity", {}).get("name", ""),
                trigger.get("kind", ""),
            ],
            "body": composed["body"],
            "cta": composed["cta"],
            "suppression_key": sk,
            "rationale": composed["rationale"],
        })

    return {"actions": actions}


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int


@app.post("/v1/reply")
def reply(body: ReplyBody):
    return handle_reply(body.conversation_id, body.merchant_id or "unknown_merchant", body.message)


@app.post("/v1/teardown")
def teardown():
    contexts.clear()
    conversations.clear()
    sent_suppression_keys.clear()
    merchant_state.clear()
    return {"status": "wiped"}
