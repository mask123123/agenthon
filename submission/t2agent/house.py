"""Executive summary (read this first)

Minimal client for the organiser-hosted House model (docs/HOUSE-MODEL.md): POST $MODEL_ENDPOINT/v1/chat/completions with
`Authorization: Bearer $MODEL_TOKEN` and `model = $MODEL_NAME`; proxies come from HTTP(S)_PROXY, which urllib reads from the
environment. The route admits 25 requests per unit and charges at admission (retries and upstream failures are not refunded),
so every request that leaves this process is counted here, a hard local cap is enforced, and three consecutive failures
open a circuit breaker. Thinking is disabled. Nothing here is ever required: callers treat any exception as "no answer".
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

ALLOWANCE = 25          # admitted requests per unit enforced by the route
MAX_OUTPUT = 4000       # output tokens per request enforced by the route


class HouseUnavailable(RuntimeError):
    pass


class House:
    def __init__(self, max_requests: int = 4, deadline_s: float = 120.0):
        self.endpoint = os.environ.get("MODEL_ENDPOINT", "").rstrip("/")
        self.token = os.environ.get("MODEL_TOKEN", "")
        self.model = os.environ.get("MODEL_NAME", "")
        self.max_requests = min(max_requests, ALLOWANCE)
        self.deadline = time.monotonic() + deadline_s
        self.used = 0
        self.fail_streak = 0

    @property
    def available(self) -> bool:
        return bool(self.endpoint and self.token and self.model)

    def chat(self, messages: list[dict], max_tokens: int = 250, timeout_s: float = 45.0) -> str:
        """One completion (temperature 0, fixed seed, thinking off). Returns the message text; raises on any failure."""
        if not self.available:
            raise HouseUnavailable("no model endpoint")
        if self.fail_streak >= 3:
            raise HouseUnavailable("circuit open after 3 consecutive failures")
        if self.used >= self.max_requests:
            raise HouseUnavailable(f"request cap {self.max_requests} reached")
        left = self.deadline - time.monotonic()
        if left < 5:
            raise HouseUnavailable("deadline reached")
        self.used += 1
        body = {"model": self.model, "messages": messages, "temperature": 0.0, "seed": 0,
                "max_tokens": min(int(max_tokens), MAX_OUTPUT), "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(self.endpoint + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=min(timeout_s, left)) as r:
                d = json.loads(r.read())
            text = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            self.fail_streak = 0 if text.strip() else self.fail_streak + 1
            if not text.strip():
                raise HouseUnavailable("empty answer")
            return text
        except HouseUnavailable:
            raise
        except Exception:
            self.fail_streak += 1
            raise
