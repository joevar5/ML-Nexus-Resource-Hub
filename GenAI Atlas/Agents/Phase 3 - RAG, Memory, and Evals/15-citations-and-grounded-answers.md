# Day 15: Citations and Grounded Answers

## Concept

A RAG answer is "grounded" if every claim in it is actually supported by the retrieved context — as opposed to the model quietly blending in unretrieved training knowledge. Grounding is not automatic just because you retrieved relevant context and put it in the prompt; the model can still ignore the context and answer from memory, especially for questions it "knows" the general answer to.

Two complementary techniques enforce grounding:

**1. Prompt-level instruction + citation requirement.** Instruct the model to cite which chunk supports each claim (e.g. `[source: chunk_3]`), and to explicitly say "not found in context" rather than guessing when the context doesn't cover something. Forcing citations changes the generation task from "answer this" to "answer this and point to where," which measurably reduces ungrounded claims — a model asked to cite is more likely to notice it can't.

**2. Post-hoc verification.** After generation, check whether each cited claim is actually supported by the chunk it cites — either with a second LLM call ("does chunk_3 support this specific claim: yes/no") or with simpler lexical/entailment checks. This catches cases where the model cited *something* but the citation doesn't actually support the claim (a "citation" that's really just decoration).

```python
def answer_with_citations(question, chunks, llm):
    numbered_context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks))
    prompt = (
        f"Context:\n{numbered_context}\n\nQuestion: {question}\n"
        "Answer using only the context. Cite the chunk number for every claim, e.g. [0]. "
        "If the context doesn't contain the answer, say so explicitly."
    )
    return llm.call(prompt)
```

The failure mode this whole lesson exists to prevent: a user reads a RAG answer with a citation next to it, trusts it *because* it has a citation, and the citation is either wrong or doesn't actually say what the answer claims it says. A citation that isn't verified is worse than no citation — it manufactures false confidence.

## Coding Problem

Write `verify_citations(answer_text: str, chunks: list[str]) -> list[dict]` that extracts citation markers like `[0]`, `[1]` from `answer_text` via regex, and for each one, returns `{"citation": i, "valid_index": bool}` — `valid_index` is `False` if the cited index is out of range for `chunks` (this catches the cheap case: a hallucinated citation number that doesn't correspond to any retrieved chunk at all, before even checking semantic support).

## Quiz

### Question 1: Grounding Isn't Automatic

**If relevant context is correctly retrieved and included in the prompt, is the resulting answer guaranteed to be grounded in that context?**

A) Yes, retrieval guarantees grounding
B) No — the model can still answer from its own training knowledge instead of the provided context, especially for questions it "knows" a general answer to
C) Yes, but only for questions under 20 words
D) Grounding is only relevant for code generation tasks

**Answer**: B

**Explanation**: Providing context doesn't force the model to use it. Without an explicit instruction (and ideally verification), a model can blend in unretrieved training knowledge, producing an answer that looks grounded but partially isn't.

### Question 2: Why Require Citations

**Why does asking a model to cite the specific chunk supporting each claim tend to reduce ungrounded answers, beyond just providing a citation?**

A) Citations are purely cosmetic and have no effect on generation
B) Forcing the model to point to a specific source changes the task from "answer" to "answer and justify," which makes it more likely to notice and flag when the context doesn't actually cover a claim
C) Citations make the API call faster
D) This technique has been shown not to work in any case

**Answer**: B

**Explanation**: Requiring an explicit pointer to supporting text changes what the model has to produce — it can't as easily gloss over a gap, because it would have to cite something that doesn't exist. This measurably improves the model's tendency to say "not found in context" instead of guessing.

### Question 3: Why Verify Citations, Not Just Require Them

**What problem does post-hoc citation verification catch that simply requiring citations in the prompt doesn't?**

A) None — requiring citations in the prompt is sufficient on its own
B) A model can produce a citation that's present but doesn't actually support the specific claim next to it — verification checks the citation is real, not just that one exists
C) Verification is only useful for detecting spelling errors
D) Citations can never be wrong once requested

**Answer**: B

**Explanation**: An instruction to cite doesn't guarantee the citation is accurate — the model can attach a plausible-looking `[3]` to a claim that chunk 3 doesn't actually support. Verification (checking the cited chunk really backs the specific claim) catches this decorative-citation failure that prompting alone can't.

## Interview Practice

**1.** A user reports that a RAG answer cited a source, but the source doesn't actually say what the answer claims. Walk through how you'd catch this class of bug in testing, before it reaches a user.

**2.** Explain to a non-technical stakeholder why "the answer has a citation" isn't the same guarantee as "the answer is correct." Use a concrete example they'd recognize as a failure.

**3.** Design a post-hoc citation verification step for a production RAG system. What would you check, how expensive is it relative to the original generation call, and would you run it on every answer or only a sample?
