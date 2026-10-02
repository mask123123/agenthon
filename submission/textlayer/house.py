"""House model client with the competition's request budget enforced locally.

Contract (track SUBMISSION_CLI.md, hub docs/HOUSE-MODEL.md): POST $MODEL_ENDPOINT/v1/chat/completions,
`Authorization: Bearer $MODEL_TOKEN`, `model = $MODEL_NAME`; proxies come from HTTP(S)_PROXY, which urllib reads
from the environment. Every request that leaves this process is counted, retries included, because the route
charges at admission and does not refund upstream failures. Three consecutive failures open a circuit breaker.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.request

ALLOWANCE = 25  # admitted requests per unit, enforced by the route
MAX_OUTPUT = 4000  # output tokens per request, enforced by the route (thinking tokens included)


class HouseUnavailable(RuntimeError):
    pass


class House:
    def __init__(self, max_requests: int = 6, deadline_s: float = 240.0):
        self.endpoint = os.environ.get("MODEL_ENDPOINT", "").rstrip("/")
        self.token = os.environ.get("MODEL_TOKEN", "")
        self.model = os.environ.get("MODEL_NAME", "")
        self.max_requests = min(max_requests, ALLOWANCE)
        self.deadline = time.monotonic() + deadline_s
        self.used = 0
        self.fail_streak = 0
        self.log: list[dict] = []

    @property
    def available(self) -> bool:
        return bool(self.endpoint and self.token and self.model)

    def _post(self, body: dict, timeout_s: float) -> dict:
        if self.fail_streak >= 3:
            raise HouseUnavailable("circuit open after 3 consecutive failures")
        if self.used >= self.max_requests:
            raise HouseUnavailable(f"request cap {self.max_requests} reached")
        left = self.deadline - time.monotonic()
        if left < 5:
            raise HouseUnavailable("text-layer deadline reached")
        self.used += 1
        req = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=min(timeout_s, left)) as r:
                d = json.loads(r.read())
            self.fail_streak = 0
            self.log.append({"ok": True, "s": round(time.monotonic() - t0, 2)})
            return d
        except Exception as e:  # charged anyway
            self.fail_streak += 1
            self.log.append({"ok": False, "s": round(time.monotonic() - t0, 2), "error": repr(e)[:160]})
            raise

    def first_token_probs(self, messages: list[dict], top: int = 10, timeout_s: float = 60.0) -> dict[str, float]:
        """Probabilities of the leading candidates for a one-token answer (thinking off).

        Uses logprobs when the route returns them; otherwise falls back to the answered token with probability 1.
        """
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": top,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        d = self._post(body, timeout_s)
        choice = d["choices"][0]
        first = ((choice.get("logprobs") or {}).get("content") or [{}])[0]
        probs = {t["token"].strip(): math.exp(t["logprob"]) for t in first.get("top_logprobs", []) if "token" in t}
        if not probs:
            ans = ((choice.get("message") or {}).get("content") or "").strip()[:1]
            probs = {ans: 1.0} if ans else {}
        return probs
