"""evaluate.py — Run the test set and score accuracy."""
import json
from pathlib import Path

from retriever import MavalRetriever
from assistant import MavalAssistant

TEST_FILE = Path(__file__).parent / "data" / "processed" / "test_set.jsonl"


def run_evaluation():
    retriever = MavalRetriever()
    assistant = MavalAssistant()

    results = {"easy": [], "medium": [], "hard": []}

    with open(TEST_FILE, "r", encoding="utf-8") as f:
        tests = [json.loads(line) for line in f if line.strip()]

    print(f"Running {len(tests)} tests...\n")

    for i, test in enumerate(tests, 1):
        q = test["question"]
        expected = test["expected_answer_contains"]
        difficulty = test["difficulty"]

        print(f"[{i}/{len(tests)}] [{difficulty.upper()}] {q}")

        try:
            chunks = retriever.search(q)
            response = assistant.answer(q, chunks)
            answer = response["answer"]

            passed = expected.lower() in answer.lower()
            status = "PASS" if passed else "FAIL"

            relevant_ids = set(test.get("relevant_chunk_ids", []))
            retrieved_ids = {c["id"] for c in chunks}
            hit_relevant = bool(relevant_ids & retrieved_ids)

            print(f"    {status} | Expected: '{expected}'")
            print(f"    Retrieved: {retrieved_ids}")
            print(f"    Relevant hit: {hit_relevant}")
            if not passed:
                print(f"    Answer: {answer[:200]}...")

            results[difficulty].append({
                "question": q,
                "passed": passed,
                "relevant_hit": hit_relevant,
                "expected": expected,
                "answer": answer,
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            results[difficulty].append({
                "question": q,
                "passed": False,
                "error": str(e),
            })

    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    total_passed = 0
    total_tests = 0

    for diff in ["easy", "medium", "hard"]:
        items = results[diff]
        if not items:
            continue
        passed = sum(1 for r in items if r.get("passed"))
        total_passed += passed
        total_tests += len(items)
        relevant_hits = sum(1 for r in items if r.get("relevant_hit"))
        print(f"\n{diff.upper()} ({len(items)} tests):")
        print(f"  Answer accuracy: {passed}/{len(items)} ({passed/len(items)*100:.0f}%)")
        print(f"  Relevant chunk retrieved: {relevant_hits}/{len(items)} ({relevant_hits/len(items)*100:.0f}%)")

    print(f"\nOVERALL: {total_passed}/{total_tests} ({total_passed/total_tests*100:.0f}%)")

    out_file = Path(__file__).parent / "data" / "processed" / "evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {out_file}")


if __name__ == "__main__":
    run_evaluation()
