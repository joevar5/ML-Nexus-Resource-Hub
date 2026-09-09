# Day 1: What is an AI Agent?

## Concept

Okay so — everyone's throwing around the word "agent" right now like it means something magical. It doesn't. Let's demystify it.

Here's the plain version: a normal LLM call is basically a vending machine. You put text in, it drops text out, transaction over. It doesn't remember, it doesn't check its work, it doesn't go do anything in the world. One shot, done.

An **agent** is what you get when you stop treating the LLM like a vending machine and start treating it like an intern with a to-do list and a phone. It can take an action, actually look at what happened, and then decide what to do *next* based on that — instead of just guessing once and walking away. That's it. That's the whole trick. No secret sauce, no special model — just a loop.

```
observe -> think -> act -> observe -> think -> act -> ... -> done
```

Round and round it goes until the goal is met (or it throws up its hands and gives up — which, relatably, does happen).

So concretely, an agent = an LLM handed three things:
1. A **goal** ("book me the cheapest flight to Austin")
2. **Tools** it's allowed to touch (search the web, call an API, run code)
3. A **loop** that keeps feeding results back in, instead of stopping after one reply

**The chatbot vs. agent gut check:** if a system tells you "the weather in Boston is 72°F" purely by regurgitating something it memorized during training — that's just a chatbot doing chatbot things. But if it actually *calls* a weather API, reads the real response, and then thinks "hm, should I check tomorrow's forecast too?" — congrats, that's an agent. The model itself might be identical. What changed is that it's now acting and reacting instead of just talking.

Here's a simple 3-point checklist. A real agent needs **all three**:

1. **Autonomy** — it decides what to do next by itself. No human wrote "then do step 2" in advance.
2. **Tool use** — it can go outside its own head and do real things (search the web, call an API, save a file).
3. **Multi-step** — it doesn't stop after one try. It takes another step, using what it just learned.

If even one of these is missing, it's not an agent — just a script pretending to be one.

**Quick example:** A basic RAG setup — search once, answer once, done — is *not* an agent. Why? Because it never checks its own work. It never asks "was that good enough, or should I search again?"

The moment you add that one question — "should I try again?" — it becomes an agent. Same tools, same search step. The only thing that changed is that it now checks and decides.

**In one line:** an agent isn't a smarter model — it's the same model, given a loop, so it can act, look, and try again instead of guessing once and stopping.
