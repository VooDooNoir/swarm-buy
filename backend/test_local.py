"""Local mock test for the Swarm Buy Lambda — no AWS credentials needed.

Monkeypatches lambda_function.converse with canned Bedrock responses, then runs
the full debate flow through lambda_handler and validates the response shape.
Run:  python3 test_local.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lambda_function as lf  # noqa: E402

CALLS = []


def fake_converse(model_id, system_text, user_text, max_tokens=700, temperature=0.7):
    CALLS.append({"model": model_id, "system_head": system_text[:60]})
    s = system_text
    if "impartial Judge" in s:
        return json.dumps({
            "winner": "Sony WH-CH720N",
            "winner_price": 148,
            "runner_up": "Anker Soundcore Space Q45",
            "verdict": "The WH-CH720N wins on comfort and ANC per dollar under the $200 cap.",
            "best_for": "Commuters and office workers who wear headphones all day.",
            "skip_if": "You're a bass-head or need 50+ hour battery.",
            "key_specs": ["Adaptive ANC", "35-hour battery", "192g lightweight"],
            "caveats": ["Plasticky build", "No LDAC support"],
            "scoreboard": {
                "audiophile": "Edifier WH950NB",
                "bargain_hunter": "Anker Soundcore Life Q30",
                "durability_nerd": "Sony WH-CH720N",
            },
        })
    if "rebuttal" in s:
        who = "Aria" if "Aria" in s else ("Mack" if "Mack" in s else "Rivet")
        return f"{who} rebuts: the rivals overpay for features nobody uses. My pick stands.\nREAFFIRM: Sony WH-CH720N"
    who = "Aria" if "Aria" in s else ("Mack" if "Mack" in s else "Rivet")
    pick = {"Aria": "Edifier WH950NB", "Mack": "Anker Soundcore Life Q30", "Rivet": "Sony WH-CH720N"}[who]
    return f"{who} argues passionately for the {pick} with catalog-grounded specs.\nCHAMPION: {pick}"


lf.converse = fake_converse


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        raise SystemExit(f"FAILED: {name} {detail}")


# 1. query parsing
cat, budget = lf.parse_query("best noise-cancelling headphones under $200")
check("parse headphones category", cat == "headphones", cat)
check("parse $200 budget", budget == 200, budget)
cat, budget = lf.parse_query("best air fryer under $150 please")
check("parse air fryer category", cat == "air_fryers", cat)
check("parse budget without $", budget == 150, budget)
cat, _ = lf.parse_query("best laptop for coding")
check("unknown category -> None", cat is None, cat)

# 2. catalog loads
catalog = lf.load_catalog()
check("catalog has 2 categories", set(catalog.keys()) >= {"headphones", "air_fryers"})
check("8 headphones", len(catalog["headphones"]) == 8, len(catalog["headphones"]))
check("8 air fryers", len(catalog["air_fryers"]) == 8, len(catalog["air_fryers"]))

# 3. full debate via handler (POST-style event)
event = {"httpMethod": "POST", "body": json.dumps({"query": "best noise-cancelling headphones under $200"})}
resp = lf.lambda_handler(event)
check("handler 200", resp["statusCode"] == 200, resp["statusCode"])
check("CORS header present", resp["headers"].get("Access-Control-Allow-Origin") == "*")
result = json.loads(resp["body"])
check("3 positions", len(result["positions"]) == 3)
check("3 rebuttals", len(result["rebuttals"]) == 3)
check("verdict winner", result["verdict"]["winner"] == "Sony WH-CH720N")
check("trailer stripped", "CHAMPION" not in result["positions"][0]["text"])
check("budget respected", result["budget"] == 200)
check("all considered under budget", all(
    p in [x["name"] for x in catalog["headphones"] if x["price"] <= 200]
    for p in result["products_considered"]
))

# 4. GET-style event
resp = lf.lambda_handler({"httpMethod": "GET", "queryStringParameters": {"q": "best air fryer under $150"}})
check("GET handler 200", resp["statusCode"] == 200, resp["statusCode"])
result = json.loads(resp["body"])
check("GET category air_fryers", result["category"] == "air_fryers", result["category"])

# 5. bad query -> 400, not 500
resp = lf.lambda_handler({"httpMethod": "POST", "body": json.dumps({"query": "best laptop for coding"})})
check("unknown category -> 400", resp["statusCode"] == 400, resp["statusCode"])

# 6. model routing: per debate 3 lite positions + 3 lite rebuttals + 1 pro judge
lite = [c for c in CALLS if c["model"] == lf.NOVA_LITE]
pro = [c for c in CALLS if c["model"] == lf.NOVA_PRO]
check("12 Nova Lite calls (2 debates x 6)", len(lite) == 12, len(lite))
check("2 Nova Pro judge calls", len(pro) == 2, len(pro))

# 7. judge JSON defensive parsing (markdown fences)
parsed = lf.parse_judge_json('```json\n{"winner": "X"}\n```')
check("fenced JSON parsed", parsed == {"winner": "X"})

print(f"\nAll checks passed. {len(CALLS)} mock Bedrock calls made.")
