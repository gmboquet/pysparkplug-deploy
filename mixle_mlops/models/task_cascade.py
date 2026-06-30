"""``TaskCascadeAdapter`` -- serve a distilled local task model (cascade) through the platform's ``/v1`` surface.

This is where the ``mixle.task`` cost loop becomes a *product*: a tiny model distilled from an expensive teacher
is hosted behind the same uniform ``ModelAdapter`` contract as any LLM, so a customer hits it through
``/v1/chat`` or ``/v1/mixle/{predict,score,decide}``. The model answers locally for ~free; when it is not
confident (a conformal prediction set that is not a singleton, or an out-of-distribution input), ``decide``
returns an **escalate** signal the gateway can route to a frontier model -- the honest, cost-cutting cascade.

It wraps a :class:`mixle.task.model.TaskModel` (always answers) or a :class:`mixle.task.calibrate.CalibratedTaskModel`
(answers-or-escalates). ``capabilities()`` advertises ``predict`` (argmax label), ``score`` (class
probabilities), ``decide`` (label or escalate), and ``chat`` (a summary), gated on what the wrapped model has.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..core.adapters import (
    ChatChunkChoice,
    ChatCompletionChunk,
    ChatRequest,
    ChoiceDelta,
    ModelAdapter,
    ModelInfo,
)


class TaskCascadeAdapter(ModelAdapter):
    """Host a distilled task model: ``predict`` (with an honest escalate signal when calibrated) and ``score``."""

    kind = "mixle"  # a distilled mixle task model (ModelInfo.kind is constrained to llm|mixle|composite)

    def __init__(self, name: str, model: Any) -> None:
        # model is a TaskModel or a CalibratedTaskModel (duck-typed: needs .batch / .adapter; .decide if calibrated)
        self._name = name
        self._model = model
        self._calibrated = hasattr(model, "decide") and hasattr(model, "task")
        self._task = model.task if self._calibrated else model
        self._labels = list(getattr(self._task.adapter, "labels", []))

    @property
    def name(self) -> str:
        return self._name

    def info(self) -> ModelInfo:
        return ModelInfo(id=self._name, kind=self.kind, capabilities=sorted(self.capabilities()))

    def capabilities(self) -> set[str]:
        # predict carries the honest escalate signal when calibrated; the mixle /decide route is a separate
        # Bayes-action-under-loss surface for numeric outcomes, which a classifier task does not implement.
        return {"chat", "predict", "score"}

    # --- distribution/decision surface (records are raw task inputs: strings or field dicts/tuples) ---
    async def predict(self, records: list[Any], **opts: Any) -> Any:
        """Local labels for each record. When calibrated, each carries the honest answer-or-``escalate`` decision.

        A confident, in-distribution input returns its label and ``escalate=False``; an ambiguous (non-singleton
        conformal set) or out-of-distribution input returns ``label=None`` and ``escalate=True`` -- the signal
        the gateway routes to a frontier model. ``escalation_rate`` is the realized fraction for this batch.
        """
        records = list(records)
        labels = self._task.batch(records)
        out: dict[str, Any] = {"model": self._name, "labels": labels}
        if self._calibrated:
            sets = self._model.predict_sets(records)
            flags = self._model._escalate_flags(records, sets)
            out["decisions"] = [
                {"label": (None if esc else label), "escalate": bool(esc), "set": list(s)}
                for label, s, esc in zip(labels, sets, flags)
            ]
            out["escalation_rate"] = float(sum(flags) / len(flags)) if len(flags) else 0.0
        return out

    async def score(self, records: list[Any], **opts: Any) -> Any:
        """Per-record class probabilities (softmax of the student's logits) over the label set."""
        proba = self._task.adapter.proba_batch(self._task.model, list(records))
        return {"model": self._name, "labels": self._labels, "proba": [[float(p) for p in row] for row in proba]}

    # --- OpenAI-compatible chat: classify the last user message and report the (honest) decision ---
    async def stream(self, req: ChatRequest) -> AsyncIterator[ChatCompletionChunk]:
        text = self._chat_summary(req)
        cid = f"chatcmpl-{self._name}"
        yield ChatCompletionChunk(
            id=cid, model=req.model, choices=[ChatChunkChoice(delta=ChoiceDelta(role="assistant"))]
        )
        yield ChatCompletionChunk(
            id=cid, model=req.model, choices=[ChatChunkChoice(delta=ChoiceDelta(content=text))]
        )
        yield ChatCompletionChunk(
            id=cid, model=req.model, choices=[ChatChunkChoice(delta=ChoiceDelta(), finish_reason="stop")]
        )

    def _chat_summary(self, req: ChatRequest) -> str:
        last = next((m for m in reversed(req.messages) if m.role == "user"), None)
        raw = (last.text().strip() if last else "")
        record = self._parse_record(raw)
        if self._calibrated:
            s = self._model.predict_set(record)
            if len(s) == 1:
                return f"[{self._name}] {s[0]}"
            return f"[{self._name}] escalate (uncertain; conformal set = {s or 'empty'})"
        return f"[{self._name}] {self._task(record)}"

    @staticmethod
    def _parse_record(raw: str) -> Any:
        """A task input may be a plain string or a JSON record (dict/list of fields)."""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (ValueError, TypeError):
            pass
        return raw


def register_demo_task_model(registry: Any, *, name: str = "demo-task") -> TaskCascadeAdapter:
    """Distill a tiny spam/ham classifier, calibrate it, and register a :class:`TaskCascadeAdapter`.

    Self-contained (needs torch via the ``mixle`` task extra): proves the serving path end to end -- a distilled
    local model answering ``predict``/``score``/``decide``/``chat`` with an honest escalate signal.
    """
    import numpy as np

    from mixle.task.calibrate import CalibratedTaskModel
    from mixle.task.distill import distill

    spam = ["free", "winner", "prize", "buy", "cheap", "offer", "click"]
    ham = ["meeting", "lunch", "project", "report", "schedule", "team", "review"]
    filler = ["the", "a", "today", "please", "thanks", "we", "you"]

    def corpus(seed: int) -> list[str]:
        r = np.random.RandomState(seed)
        out = []
        for words in (spam, ham):
            for _ in range(120):
                toks = list(r.choice(words, size=2)) + list(r.choice(filler, size=r.randint(3, 6)))
                r.shuffle(toks)
                out.append(" ".join(toks))
        r.shuffle(out)
        return out

    def teacher(texts: list[str]) -> list[str]:
        s = set(spam)
        return ["spam" if any(w in t.split() for w in s) else "ham" for t in texts]

    train, cal = corpus(1), corpus(2)
    student = distill(teacher, train, n=4, dim=256, hidden=[32], epochs=150, seed=0, task="spam vs ham")
    model = CalibratedTaskModel(student, alpha=0.1).calibrate(cal, teacher(cal))
    adapter = TaskCascadeAdapter(name, model)
    registry.register(adapter)
    return adapter
