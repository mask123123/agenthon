"""House model client with the competition's budget enforced locally.

Contract (SUBMISSION_CLI.md): POST $MODEL_ENDPOINT/v1/chat/completions, bearer $MODEL_TOKEN,
model $MODEL_NAME; proxies come from HTTP(S)_PROXY (urllib reads them from the environment).
Every request that leaves this process counts against the per-unit allowance, retries included,
because the route charges at admission and does not refund upstream failures.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

ALLOWANCE = 25  # admitted requests per unit (route-enforced)
MAX_OUTPUT = 4000  # output tokens per request (route-enforced), thinking included


class BudgetExhausted(RuntimeError):
    pass


class House:
    def __init__(self, max_requests: int = 14, deadline_s: float = 900.0, timeout_s: float = 240.0):
        self.endpoint = os.environ.get("MODEL_ENDPOINT", "").rstrip("/")
        self.token = os.environ.get("MODEL_TOKEN", "")
        self.model = os.environ.get("MODEL_NAME", "")
        self.max_requests = min(max_requests, ALLOWANCE)
        self.deadline = time.monotonic() + deadline_s
        self.timeout_s = timeout_s
        self.used = 0
        self.fail_streak = 0  # circuit breaker: stop calling after 3 consecutive failures
        self.log: list[dict] = []
        self._lock = threading.Lock()
        self._extra: dict = {}
        self._last: dict | None = None

    @property
    def available(self) -> bool:
        return bool(self.endpoint and self.token and self.model)

    def first_token_probs(self, messages: list[dict], top: int = 10, timeout_s: float = 60.0) -> dict[str, float]:
        """Probabilities of the top candidates for the first answer token (thinking off, one token)."""
        self._extra = {"logprobs": True, "top_logprobs": top}
        try:
            self.chat(messages, max_tokens=1, thinking=False, timeout_s=timeout_s)
        finally:
            self._extra = {}
        lp = (self._last or {}).get("logprobs") or {}
        first = (lp.get("content") or [{}])[0]
        import math
        return {t["token"].strip(): math.exp(t["logprob"]) for t in first.get("top_logprobs", [])}

    def chat(self, messages: list[dict], *, max_tokens: int = 1500, thinking: bool = False,
             temperature: float = 0.0, timeout_s: float | None = None) -> str:
        with self._lock:
            if self.fail_streak >= 3:
                raise BudgetExhausted("circuit open after 3 consecutive failures")
            if self.used >= self.max_requests:
                raise BudgetExhausted(f"request cap {self.max_requests} reached")
            self.used += 1
        left = self.deadline - time.monotonic()
        if left < 20:
            raise TimeoutError("unit deadline reached")
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "seed": 0,
            "max_tokens": min(max_tokens, MAX_OUTPUT),
            "chat_template_kwargs": {"enable_thinking": thinking},
            **self._extra,
        }
        req = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        t0 = time.monotonic()
        entry = {"thinking": thinking, "max_tokens": body["max_tokens"]}
        try:
            with urllib.request.urlopen(req, timeout=min(timeout_s or self.timeout_s, left - 10)) as r:
                d = json.loads(r.read())
            msg = d["choices"][0]["message"]
            self._last = d["choices"][0]
            entry.update(ok=True, usage=d.get("usage"), finish=d["choices"][0].get("finish_reason"))
            self.fail_streak = 0
            return msg.get("content") or ""
        except Exception as e:  # charged anyway
            entry.update(ok=False, error=repr(e)[:200])
            self.fail_streak += 1
            raise
        finally:
            entry["seconds"] = round(time.monotonic() - t0, 2)
            self.log.append(entry)


def parse_json(text: str):
    """First JSON object in a reply, tolerating code fences, prose around it and trailing commas."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    start = t.find("{")
    while start != -1:
        depth, instr, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if instr:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    instr = False
                continue
            if c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = re.sub(r",\s*([}\]])", r"\1", t[start : i + 1])
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        break
        start = t.find("{", start + 1)
    return None
