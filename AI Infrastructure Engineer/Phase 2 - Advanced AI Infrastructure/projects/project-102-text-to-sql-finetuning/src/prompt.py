"""One prompt format for baseline, training, evaluation and serving. Change it in one place only."""
import re


def build_prompt(context: str, question: str) -> str:
    """The text the model reads. It must end right where the SQL should start."""
    return f"### Schema\n{context.strip()}\n\n### Question\n{question.strip()}\n\n### SQL\n"


def extract_sql(generation: str) -> str:
    """Pull one SQL statement out of raw model text, which often has fences, prose or extra statements."""
    text = generation.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    text = text.split("###")[0]          # the model started a new prompt section
    text = text.split(";")[0]            # keep the first statement only
    return "\n".join(l for l in text.strip().splitlines() if l.strip()).strip()
