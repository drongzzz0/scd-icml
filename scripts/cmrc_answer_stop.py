"""Shared stop policy for the native and SCD arms of the CMRC replay."""

import time


class CMRCAnswerStop:
    def __init__(self, tokenizer, extract_answer, context, separators, stop_ids, max_new_tokens):
        self.tokenizer = tokenizer
        self.extract_answer = extract_answer
        self.context = context
        self.separators = tuple(separators)
        self.stop_ids = frozenset(stop_ids)
        self.max_new_tokens = max_new_tokens
        self.observed = ()
        self.prediction = ""
        self.reason = None

    def __call__(self, token_ids):
        ids = tuple(token_ids)
        if self.reason is not None or ids[:-1] != self.observed or len(ids) != len(self.observed) + 1:
            raise ValueError("CMRC observer requires one new token per call until it stops")
        self.observed = ids
        if ids[-1] in self.stop_ids:
            self.reason = "stop_token"
            return self.reason

        raw = self.tokenizer.decode(ids, skip_special_tokens=False).lstrip()
        visible = self.tokenizer.decode(ids, skip_special_tokens=True).lstrip()
        self.prediction = self.extract_answer(visible, context=self.context)
        if visible and any(sep and (sep in raw or sep in visible) for sep in self.separators):
            self.reason = "text_boundary"
        else:
            plain_answer = self.extract_answer(visible)
            if plain_answer and self.prediction and self.prediction != plain_answer:
                self.reason = "context_span_boundary"
            elif len(ids) >= self.max_new_tokens:
                self.reason = "max_new_tokens"
        return self.reason


class TimedAnswerStop:
    """Measure from the request call to the first output and stop decision."""

    def __init__(self, stop):
        self.stop = stop
        self.started_at = time.perf_counter()
        self.first_token_ms = None
        self.completion_ms = None

    def __call__(self, token_ids):
        observed_at = time.perf_counter()
        if self.first_token_ms is None:
            self.first_token_ms = (observed_at - self.started_at) * 1000
        reason = self.stop(token_ids)
        if reason is not None:
            self.completion_ms = (time.perf_counter() - self.started_at) * 1000
        return reason


def measure_answer_request(engine, request, stop):
    """Observe target submission at the same engine entry for both methods."""
    add_request = engine.add_request
    original_override = vars(engine).get("add_request")
    submitted_ms = None

    def record_submit(*args, **kwargs):
        nonlocal submitted_ms
        if submitted_ms is not None:
            raise RuntimeError("expected one target request per answer")
        submitted_ms = (time.perf_counter() - timed.started_at) * 1000
        return add_request(*args, **kwargs)

    engine.add_request = record_submit
    try:
        timed = TimedAnswerStop(stop)
        result = request(timed)
        returned_ms = (time.perf_counter() - timed.started_at) * 1000
    finally:
        if original_override is None:
            del engine.add_request
        else:
            engine.add_request = original_override

    ttft_ms, completion_ms = timed.first_token_ms, timed.completion_ms
    if any(value is None for value in (submitted_ms, ttft_ms, completion_ms)):
        raise RuntimeError("missing target submission, first token or stop decision")
    if not 0 <= submitted_ms <= ttft_ms <= completion_ms <= returned_ms:
        raise RuntimeError("request timestamps are out of order")
    return result, {
        "pre_target_submit_ms": submitted_ms,
        "target_to_first_token_ms": ttft_ms - submitted_ms,
        "decode_after_first_ms": completion_ms - ttft_ms,
        "stop_to_return_ms": returned_ms - completion_ms,
        "ttft_ms": ttft_ms,
        "completion_ms": completion_ms,
        "request_wall_ms": returned_ms,
    }
