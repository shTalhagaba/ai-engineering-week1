import asyncio
import json
from pathlib import Path

from rag_answer import ask
from rag_store import search

TOP_K = 4


async def main():
    cases = json.loads(Path("eval_set.json").read_text())
    stats = {"retrieval": [0, 0], "answer": [0, 0], "refusal": [0, 0]}
    failures = []

    for case in cases:
        q = case["question"]
        result = await ask(q, top_k=TOP_K)

        if case["in_scope"]:
            # 1. Retrieval: sahi page top-k mein aaya?
            hits = await search(q, TOP_K)
            got_pages = {c["page"] for c, _ in hits}
            ok_ret = bool(got_pages & set(case["expected_pages"]))
            stats["retrieval"][0] += ok_ret
            stats["retrieval"][1] += 1

            # 2. Answer: zaroori lafz jawab mein hain?
            text = result.answer.lower()
            ok_ans = result.found_in_documents and all(
                k.lower() in text for k in case["must_include"]
            )
            stats["answer"][0] += ok_ans
            stats["answer"][1] += 1

            if not (ok_ret and ok_ans):
                failures.append((q, f"retrieval={ok_ret}, answer={ok_ans}", result.answer))
        else:
            # 3. Refusal: out-of-scope par "nahi mila" aana chahiye
            ok_ref = not result.found_in_documents
            stats["refusal"][0] += ok_ref
            stats["refusal"][1] += 1
            if not ok_ref:
                failures.append((q, "should have refused", result.answer))

        await asyncio.sleep(1)  # rate limit se bachne ke liye

    print("\n=== Results ===")
    for name, (ok, total) in stats.items():
        if total:
            print(f"{name:10s}: {ok}/{total}  ({ok / total:.0%})")

    if failures:
        print("\n=== Failures ===")
        for q, why, ans in failures:
            print(f"\nQ: {q}\n  Why: {why}\n  Answer: {ans[:200]}")


if __name__ == "__main__":
    asyncio.run(main())