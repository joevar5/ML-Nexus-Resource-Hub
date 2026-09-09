# Day 9: Retries and Fallbacks

## Concept

Retrying is not "call it again" — done naively it makes outages worse. Three things a real retry policy needs:

**1. Exponential backoff with jitter.** Retrying immediately after a failure hammers a service that's already struggling (or hits the same rate limit again instantly). Backoff spaces retries out (1s, 2s, 4s, 8s...); jitter (randomizing that delay slightly) prevents many clients from retrying in lockstep and re-causing the spike that failed them in the first place.

```python
import random, time

def retry_with_backoff(fn, max_attempts=4, base_delay=1.0):
    for attempt in range(max_attempts):
        try:
            return fn()
        except RetryableError:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(delay)
```

**2. A cap on attempts.** Unbounded retry is just a slow way to hang. Cap it, and when the cap is hit, surface a clear failure rather than continuing to try silently.

**3. A fallback path for when retries are exhausted.** Not every failure should end the task. Common fallbacks: try an alternate tool that does something similar (a backup search API), degrade to a simpler answer ("I couldn't verify this live, but based on my training data..."), or ask a human. The fallback should be a decision made deliberately, not the retry loop just... stopping.

The mistake to avoid: wrapping *every* tool call in the same generic retry policy. A call that mutates state (sends an email, charges a card) is not safe to blindly retry — a timeout doesn't tell you whether the action happened before or after the timeout. Retries are safe by default only for idempotent operations (reads, or writes designed to be safely repeatable, e.g. via an idempotency key). Mutating, non-idempotent calls need either an idempotency key or a "check first" step before retrying.

## Coding Problem

Write `retry_with_backoff(fn, max_attempts, base_delay, is_idempotent)` that behaves as in the code above, but raises immediately (no retry) if `is_idempotent` is `False` and the exception occurred **after** partial execution began (simulate this with a `PartialExecutionError` subclass of `RetryableError` that should never be retried even for idempotent-looking calls, since "partial execution" means state may have already changed). Only retry plain `RetryableError` instances.

## Quiz

### Question 1: Why Backoff and Jitter

**Why is "retry immediately" a bad default policy for a failed API call?**

A) Immediate retries are technically impossible
B) It can worsen an already-struggling service, and if many clients retry in lockstep it can recreate the exact load spike that caused the failure
C) APIs charge extra for fast retries
D) There's no downside — immediate retry is always best

**Answer**: B

**Explanation**: A service failing under load doesn't benefit from instant re-hammering. Exponential backoff spaces out retries, and jitter prevents synchronized retry storms from many clients hitting the service at the same moment.

### Question 2: Idempotency

**Why is it dangerous to blindly retry a tool call that sends an email or charges a payment?**

A) It's never dangerous — retries are always safe
B) A timeout doesn't reveal whether the action completed before or after the failure, so a naive retry risks duplicating a real-world side effect (sending twice, charging twice)
C) Email and payment APIs don't support retries
D) It's only a concern for read operations

**Answer**: B

**Explanation**: Non-idempotent operations can already have taken effect by the time a timeout is observed. Retrying without an idempotency key or a "did this already happen?" check risks a duplicate side effect — this is a fundamentally different risk than retrying a pure read.

### Question 3: Fallback Design

**Why should a fallback path be a deliberate decision rather than just "the retry loop giving up"?**

A) Because retries should never have a limit
B) Because "giving up" with no fallback leaves the agent (and the task) stuck with no useful outcome; a deliberate fallback (alternate tool, degraded answer, human handoff) gives the task somewhere to go
C) Fallbacks are only relevant for UI code, not agents
D) There's no meaningful distinction between the two

**Answer**: B

**Explanation**: An exhausted retry loop with no plan for what happens next just fails silently or crashes. Designing an explicit fallback — a different tool, a lower-confidence answer, escalation to a human — turns an inevitable failure mode into a handled one.

## Interview Practice

**1.** Design the retry policy for an agent that sends transactional emails via a third-party API. Specifically address: what's retryable, what's not, and how you prevent a duplicate email if a request times out but may have actually succeeded.

**2.** A teammate proposes retrying every failed tool call up to 10 times with no backoff, "to maximize the chance of success." Explain what actually goes wrong with this policy under real failure conditions.

**3.** Compare a fallback that degrades to a simpler answer versus one that escalates to a human. Describe a scenario for each where it's clearly the right choice, and one where using the wrong one would be a real problem.
