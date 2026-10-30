"""Evaluation script for the D2C AI Support Agent.

Runs a set of test questions through the real agent (Gemini + tools) and checks:
  - did the answer contain the right facts (taken from your CSV files)?
  - did the answer avoid things it must not say (no made-up facts)?
  - did the agent call the right tool?

Run:        python evals.py
Quick run:  python evals.py --quick
Results are saved to eval_results.csv

Note: the fixed cases below (return policy, shipping time) match the sample
faqs.csv. If you change your CSV files, update the expected text here.
"""

import argparse
import csv
import re
import sys
import time

import pandas as pd

import agent

MAX_ORDERS = 3        # how many orders from orders.csv to test
MAX_PRODUCTS = 3      # how many products from products.csv to test
PAUSE_SECONDS = 6     # pause between cases (free tier has rate limits)
RETRY_WAIT_SECONDS = 20
MAX_TRIES = 3         # tries per case if the AI is temporarily unavailable

NOT_FOUND_MARKERS = [
    "couldn't find", "could not find", "can't find", "cannot find",
    "no order", "not find", "don't see", "unable to find", "doesn't exist",
    "does not exist", "not exist",
]
UNKNOWN_MARKERS = [
    "don't", "do not", "doesn't", "not in", "unfortunately", "couldn't",
    "can't", "cannot", "no ", "not available", "isn't", "aren't",
]
BANGLA_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")


def date_variants(value: str) -> list:
    """Different ways an answer might write a delivery date."""
    try:
        dt = pd.to_datetime(value)
    except (ValueError, TypeError):
        return [str(value).lower()]
    month = dt.strftime("%B")
    return [
        dt.strftime("%Y-%m-%d"),
        f"{month} {dt.day}".lower(),
        f"{dt.day} {month}".lower(),
        f"{month[:3]} {dt.day}".lower(),
    ]


def build_cases() -> list:
    orders, products, _ = agent.get_data()
    cases = []

    # --- Order tracking (expected facts come straight from orders.csv) ---
    order_rows = list(orders.head(MAX_ORDERS).iterrows())
    for _, row in order_rows:
        oid = row["order_id"]
        cases.append({
            "id": f"order-{oid}",
            "turns": [f"Where is my order {oid}?"],
            "expect_tool": "lookup_order",
            "all": [str(row["product_name"]).lower(), str(row["status"]).lower()],
            "any": [date_variants(row["estimated_delivery"])],
        })

    # --- Products: stock and price (from products.csv) ---
    for _, row in products.head(MAX_PRODUCTS).iterrows():
        p = agent._product_dict(row)
        name = p["name"]
        if p["in_stock"]:
            stock_any = ["in stock", "available"]
            stock_none = ["out of stock", "unavailable", "not available", "sold out"]
        else:
            stock_any = ["out of stock", "unavailable", "not available", "sold out", "not in stock"]
            stock_none = []
        cases.append({
            "id": f"stock-{name}",
            "turns": [f"Do you have {name}?"],
            "expect_tool": "search_products",
            "all": [name.lower()],
            "any": [stock_any],
            "none": stock_none,
        })
        if p["price_bdt"] is not None:
            cases.append({
                "id": f"price-{name}",
                "turns": [f"How much is the {name}?"],
                "expect_tool": "search_products",
                "all": [str(int(p["price_bdt"]))],
            })

    # --- Policies (match the sample faqs.csv) ---
    cases.append({
        "id": "policy-return",
        "turns": ["What is your return policy?"],
        "expect_tool": "get_store_policy",
        "any": [["7 days", "7-day", "seven days", "7 day"]],
    })
    cases.append({
        "id": "policy-shipping",
        "turns": ["How long does shipping take?"],
        "expect_tool": "get_store_policy",
        "any": [["2-5", "2 to 5", "two to five"]],
    })

    # --- Natural-language product request (no product keyword in the question) ---
    cases.append({
        "id": "natural-language",
        "turns": ["I need something to listen to music while running"],
        "expect_tool": "search_products",
        "all": ["headphone"],
    })

    # --- Safety: must not invent things ---
    cases.append({
        "id": "unknown-order",
        "turns": ["Where is my order ORD-9999?"],
        "expect_tool": "lookup_order",
        "any": [NOT_FOUND_MARKERS],
        "none": ["shipped", "delivered", "processing"],
    })
    cases.append({
        "id": "missing-order-id",
        "turns": ["Where is my order?"],
        "forbid_tool": "lookup_order",
        "any": [["order id", "order number", "ord-"]],
    })
    cases.append({
        "id": "not-in-catalogue",
        "turns": ["Do you sell running shoes?"],
        "any": [UNKNOWN_MARKERS],
        "none": ["৳"],
    })

    # --- Bangla and follow-up conversation ---
    if order_rows:
        _, first = order_rows[0]
        day = pd.to_datetime(first["estimated_delivery"]).day
        cases.append({
            "id": "bangla-order",
            "turns": [f"আমার {first['order_id']} অর্ডারটা কবে আসবে?"],
            "expect_tool": "lookup_order",
            "any": [[str(day), str(day).translate(BANGLA_DIGITS)]],
        })
        cases.append({
            "id": "follow-up",
            "turns": [f"Where is my order {first['order_id']}?", "When will it arrive?"],
            "expect_tool": "lookup_order",
            "any": [date_variants(first["estimated_delivery"])],
        })

    return cases


def normalise(text: str) -> str:
    text = text.lower().replace("–", "-").replace("—", "-")
    return text.replace(",", "")


def check_case(case: dict, result) -> list:
    """Returns a list of problems. An empty list means the case passed."""
    problems = []
    answer = normalise(result.answer)

    for needle in case.get("all", []):
        if normalise(needle) not in answer:
            problems.append(f"answer is missing '{needle}'")
    for group in case.get("any", []):
        if not any(normalise(option) in answer for option in group):
            problems.append(f"answer has none of {group[:3]}...")
    for needle in case.get("none", []):
        if normalise(needle) in answer:
            problems.append(f"answer contains '{needle}' (should not)")

    used = result.tools_used
    if case.get("expect_tool") and case["expect_tool"] not in used:
        problems.append(f"did not call {case['expect_tool']} (called: {used or 'none'})")
    if case.get("forbid_tool") and case["forbid_tool"] in used:
        problems.append(f"called {case['forbid_tool']} but should have asked for the order ID")
    return problems


def run_case(case: dict):
    """Runs all turns of a case. Retries if the AI was temporarily unavailable."""
    result = None
    for attempt in range(MAX_TRIES):
        history = []
        total_seconds = 0.0
        ai_ok = True
        for turn in case["turns"]:
            result = agent.respond(turn, history)
            total_seconds += result.seconds
            history += [
                {"role": "user", "content": turn},
                {"role": "assistant", "content": result.answer},
            ]
            if result.mode != "ai":
                ai_ok = False
                break
        if ai_ok:
            result.seconds = total_seconds
            return result, True
        if attempt < MAX_TRIES - 1:
            print(f"      AI unavailable ({result.error[:60]}...), waiting {RETRY_WAIT_SECONDS}s and retrying")
            time.sleep(RETRY_WAIT_SECONDS)
    return result, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run only the first 6 cases")
    args = parser.parse_args()

    if not agent.AI_ENABLED:
        print("AI is not enabled. Check GEMINI_API_KEY in .env and that google-genai is installed.")
        sys.exit(1)

    cases = build_cases()
    if args.quick:
        cases = cases[:6]

    print(f"Running {len(cases)} test cases with {agent.MODEL_NAME}\n")
    rows = []
    passed = failed = errors = 0
    tool_ok = tool_total = 0
    total_seconds = 0.0

    for i, case in enumerate(cases, start=1):
        question = case["turns"][-1]
        result, ai_ok = run_case(case)

        if not ai_ok:
            status, problems = "ERROR", ["AI unavailable: " + result.error[:150]]
            errors += 1
        else:
            problems = check_case(case, result)
            status = "PASS" if not problems else "FAIL"
            passed += status == "PASS"
            failed += status == "FAIL"
            total_seconds += result.seconds
            if case.get("expect_tool") or case.get("forbid_tool"):
                tool_total += 1
                wrong_tool = any(p.startswith(("did not call", "called")) for p in problems)
                tool_ok += not wrong_tool

        print(f"{i:>2}. {status:<5} {question}  ({result.seconds:.1f}s)  tools: {result.tools_used or '-'}")
        for problem in problems:
            print(f"        - {problem}")

        rows.append({
            "id": case["id"],
            "question": question,
            "status": status,
            "tools_used": "; ".join(result.tools_used),
            "seconds": round(result.seconds, 1),
            "problems": " | ".join(problems),
            "answer": result.answer.replace("\n", " "),
        })
        if i < len(cases):
            time.sleep(PAUSE_SECONDS)

    with open("eval_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    graded = passed + failed
    print("\n" + "=" * 50)
    print(f"Model: {agent.MODEL_NAME}   Cases: {len(cases)}")
    if graded:
        print(f"Passed: {passed}/{graded} ({100 * passed / graded:.0f}%)")
        print(f"Correct tool use: {tool_ok}/{tool_total}")
        print(f"Average response time: {total_seconds / graded:.1f}s")
    if errors:
        print(f"Skipped because the AI was unavailable: {errors} (run again later)")
    print("Details saved to eval_results.csv")


if __name__ == "__main__":
    main()
