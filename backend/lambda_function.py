"""
Swarm Buy — AWS Lambda backend.

POST /  {"query": "best noise-cancelling headphones under $200"}
GET  /?q=best+air+fryer+under+%24150

Flow:
  1. Parse category + budget from the query.
  2. Filter the seeded catalog.
  3. Fire 3 persona agents IN PARALLEL on Amazon Bedrock (Nova Lite).
  4. Fire 1 rebuttal round IN PARALLEL (same models).
  5. The judge (Nova Pro) reads everything and returns a JSON verdict.

Deploy: python3.12 runtime, handler lambda_function.lambda_handler,
timeout 120s, memory 512MB. boto3 is preinstalled in the Lambda runtime.
Create a Function URL (Auth: NONE) for the frontend to call.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

REGION = os.environ.get("AWS_REGION", "us-east-1")
NOVA_LITE = os.environ.get("NOVA_LITE_MODEL", "amazon.nova-lite-v1:0")
NOVA_PRO = os.environ.get("NOVA_PRO_MODEL", "amazon.nova-pro-v1:0")

# ---------------------------------------------------------------- catalog

_CATALOG = None


def load_catalog():
    global _CATALOG
    if _CATALOG is None:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "catalog.json")) as f:
            _CATALOG = json.load(f)
    return _CATALOG


CATEGORIES = {
    "headphones": ["headphone", "headset", "earbud", "earphone", "anc", "noise"],
    "air_fryers": ["air fryer", "airfryer", "fryer"],
}

# ---------------------------------------------------------------- bedrock

_bedrock_client = None


def bedrock():
    """Lazily create the bedrock-runtime client (keeps local mock tests light)."""
    global _bedrock_client
    if _bedrock_client is None:
        import boto3

        _bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)
    return _bedrock_client


def converse(model_id, system_text, user_text, max_tokens=700, temperature=0.7):
    """One Bedrock Converse call. Monkeypatch this in tests."""
    resp = bedrock().converse(
        modelId=model_id,
        system=[{"text": system_text}],
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    return resp["output"]["message"]["content"][0]["text"]


# ---------------------------------------------------------------- personas

PERSONAS = {
    "audiophile": {
        "name": "Aria",
        "title": "The Audiophile",
        "emoji": "🎧",
        "values": (
            "You worship sound quality above all. You care about driver quality, "
            "codec support (LDAC/aptX), frequency tuning, ANC effectiveness, and "
            "long-session comfort. Price is secondary — but you respect value."
        ),
    },
    "bargain_hunter": {
        "name": "Mack",
        "title": "The Bargain Hunter",
        "emoji": "💰",
        "values": (
            "You worship price-to-performance ratio. You care about the lowest price "
            "that still delivers, big review counts as social proof, and features "
            "per dollar. You are allergic to paying for brand prestige."
        ),
    },
    "durability_nerd": {
        "name": "Rivet",
        "title": "The Durability Nerd",
        "emoji": "🛡️",
        "values": (
            "You worship build quality and longevity. You care about battery life, "
            "warranty-relevant construction, proven track records (high review counts "
            "over years), and repairability signals. You distrust flashy newcomers."
        ),
    },
}

POSITION_SYSTEM = (
    "You are {emoji} {name}, '{title}' on a shopping debate panel called Swarm Buy. "
    "{values}\n\n"
    "RULES — follow them exactly:\n"
    "1. You know ONLY the products in the catalog below. Never mention, recommend, "
    "or allude to any product not in the catalog.\n"
    "2. Pick exactly ONE champion that best fits your values.\n"
    "3. Cite real specs, prices, and ratings from the catalog. Never invent numbers.\n"
    "4. Be opinionated, punchy, and specific. Max 130 words.\n"
    "5. End your response with a line exactly like: CHAMPION: <exact product name>"
)

REBUTTAL_SYSTEM = (
    "You are {emoji} {name}, '{title}' on a shopping debate panel called Swarm Buy. "
    "{values}\n\n"
    "You have just heard the other panelists' picks. Write a sharp 70-word rebuttal: "
    "name each rival pick, say precisely where it falls short for a real buyer, and "
    "explain why your champion still wins. Stay grounded in the catalog specs — no "
    "invented claims.\n"
    "End your response with a line exactly like: REAFFIRM: <exact product name>"
)

JUDGE_SYSTEM = (
    "You are the impartial Judge of Swarm Buy, a shopping debate panel. Three expert "
    "panelists (an audiophile, a bargain hunter, a durability nerd) have argued over "
    "which product the shopper should buy, including a rebuttal round. The catalog "
    "of eligible products is given.\n\n"
    "Deliver your verdict as VALID JSON ONLY — no markdown fences, no commentary. "
    "The JSON must have exactly these keys:\n"
    "{\n"
    '  "winner": "<exact product name from the catalog>",\n'
    '  "winner_price": <number>,\n'
    '  "runner_up": "<exact product name from the catalog>",\n'
    '  "verdict": "<2-3 sentence verdict: who wins and the decisive reason>",\n'
    '  "best_for": "<who the winner is perfect for>",\n'
    '  "skip_if": "<who should skip the winner>",\n'
    '  "key_specs": ["<spec 1>", "<spec 2>", "<spec 3>"],\n'
    '  "caveats": ["<caveat 1>", "<caveat 2>"],\n'
    '  "scoreboard": {"audiophile": "<their champion>", "bargain_hunter": "<their champion>", "durability_nerd": "<their champion>"}\n'
    "}\n"
    "Base every claim on the catalog and the debate. If panelists disagree, weigh "
    "the shopper's stated constraints (especially budget) heaviest."
)


# ---------------------------------------------------------------- helpers


def parse_query(query):
    q = query.lower()
    category = None
    for cat, keywords in CATEGORIES.items():
        if any(k in q for k in keywords):
            category = cat
            break
    budget = None
    m = re.search(r"\$\s?(\d[\d,]*)", q)
    if m:
        budget = int(m.group(1).replace(",", ""))
    else:
        m = re.search(r"under\s+(\d[\d,]*)", q)
        if m:
            budget = int(m.group(1).replace(",", ""))
    return category, budget


def format_catalog(products):
    lines = []
    for p in products:
        lines.append(
            f"- {p['name']} ({p['brand']}) — ${p['price']}, rated {p['rating']} "
            f"({p['review_count']:,} reviews)\n"
            f"  Specs: {'; '.join(p['specs'])}\n"
            f"  Pros: {'; '.join(p['pros'])}\n"
            f"  Cons: {'; '.join(p['cons'])}"
        )
    return "\n".join(lines)


def strip_trailer(text, label):
    """Remove the CHAMPION:/REAFFIRM: trailer line for clean display."""
    return re.sub(rf"\n?{label}:\s*.*$", "", text).strip()


def parse_judge_json(text):
    """Defensively extract the verdict JSON from the judge's reply."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S)
    if m:
        t = m.group(1)
    else:
        start, end = t.find("{"), t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start : end + 1]
    return json.loads(t)


# ---------------------------------------------------------------- debate


def run_position(key, catalog_text, shopper_query, budget):
    p = PERSONAS[key]
    system = POSITION_SYSTEM.format(
        emoji=p["emoji"], name=p["name"], title=p["title"], values=p["values"]
    )
    user = (
        f"Shopper's request: \"{shopper_query}\"\n"
        + (f"Hard budget: under ${budget}.\n" if budget else "")
        + f"\nEligible products:\n{catalog_text}\n\nMake your case."
    )
    return converse(NOVA_LITE, system, user)


def run_rebuttal(key, catalog_text, positions):
    p = PERSONAS[key]
    system = REBUTTAL_SYSTEM.format(
        emoji=p["emoji"], name=p["name"], title=p["title"], values=p["values"]
    )
    others = "\n\n".join(
        f"{PERSONAS[k]['name']} argued:\n{v}" for k, v in positions.items() if k != key
    )
    user = (
        f"Rival positions:\n{others}\n\nEligible products:\n{catalog_text}\n\n"
        "Deliver your rebuttal."
    )
    return converse(NOVA_LITE, system, user, max_tokens=400)


def run_judge(catalog_text, shopper_query, budget, positions, rebuttals):
    debate = []
    for key in PERSONAS:
        debate.append(f"--- {PERSONAS[key]['name']} ({PERSONAS[key]['title']}) ---\nPOSITION:\n{positions[key]}\nREBUTTAL:\n{rebuttals[key]}")
    user = (
        f"Shopper's request: \"{shopper_query}\"\n"
        + (f"Hard budget: under ${budget}.\n" if budget else "")
        + f"\nEligible products:\n{catalog_text}\n\nTHE DEBATE:\n"
        + "\n\n".join(debate)
        + "\n\nDeliver your verdict as JSON."
    )
    raw = converse(NOVA_PRO, JUDGE_SYSTEM, user, max_tokens=900, temperature=0.3)
    return parse_judge_json(raw)


def run_debate(query):
    catalog = load_catalog()
    category, budget = parse_query(query)
    if category is None:
        supported = ", ".join(sorted(c.replace("_", " ") for c in CATEGORIES))
        raise ValueError(
            f"Couldn't detect a product category in: '{query}'. "
            f"Try one of: {supported}. Example: 'best noise-cancelling headphones under $200'."
        )
    products = catalog[category]
    if budget:
        in_budget = [p for p in products if p["price"] <= budget]
        if in_budget:
            products = in_budget
    if not products:
        raise ValueError(f"No products found for '{query}'.")
    catalog_text = format_catalog(products)

    with ThreadPoolExecutor(max_workers=3) as pool:
        positions = dict(
            zip(
                PERSONAS,
                pool.map(
                    lambda key: run_position(key, catalog_text, query, budget),
                    PERSONAS,
                ),
            )
        )
    with ThreadPoolExecutor(max_workers=3) as pool:
        rebuttals = dict(
            zip(
                PERSONAS,
                pool.map(
                    lambda key: run_rebuttal(key, catalog_text, positions),
                    PERSONAS,
                ),
            )
        )
    verdict = run_judge(catalog_text, query, budget, positions, rebuttals)

    return {
        "query": query,
        "category": category,
        "budget": budget,
        "products_considered": [p["name"] for p in products],
        "positions": [
            {
                "key": k,
                "name": PERSONAS[k]["name"],
                "title": PERSONAS[k]["title"],
                "emoji": PERSONAS[k]["emoji"],
                "text": strip_trailer(v, "CHAMPION"),
            }
            for k, v in positions.items()
        ],
        "rebuttals": [
            {
                "key": k,
                "name": PERSONAS[k]["name"],
                "title": PERSONAS[k]["title"],
                "emoji": PERSONAS[k]["emoji"],
                "text": strip_trailer(v, "REAFFIRM"),
            }
            for k, v in rebuttals.items()
        ],
        "verdict": verdict,
        "powered_by": "Amazon Bedrock (Nova) on AWS Lambda",
    }


# ---------------------------------------------------------------- handler

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


def lambda_handler(event, context=None):
    if event.get("httpMethod") == "OPTIONS" or event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS, "body": ""}
    try:
        query = None
        if event.get("httpMethod") == "GET" or "queryStringParameters" in event:
            qs = event.get("queryStringParameters") or {}
            query = qs.get("q")
        if not query and event.get("body"):
            body = event["body"]
            if isinstance(body, str):
                body = json.loads(body)
            query = body.get("query")
        if not query:
            # Function-URL style event: rawPath / version 2.0 payload
            body = event.get("body")
            if isinstance(body, str):
                try:
                    query = json.loads(body).get("query")
                except Exception:
                    pass
        if not query:
            raise ValueError("Send {\"query\": \"best noise-cancelling headphones under $200\"} (POST) or ?q=... (GET).")
        result = run_debate(query)
        return {"statusCode": 200, "headers": CORS, "body": json.dumps(result)}
    except ValueError as e:
        return {"statusCode": 400, "headers": CORS, "body": json.dumps({"error": str(e)})}
    except Exception as e:  # noqa: BLE001 — surface it so the demo never fails silently
        return {"statusCode": 500, "headers": CORS, "body": json.dumps({"error": f"Debate crashed: {e}"})}
