"""API tests for NLP Well Report Analyzer."""

import sys
import json
import time
from fastapi.testclient import TestClient

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))

from app import app

PASS = 0
FAIL = 0
client = TestClient(app)


def test(name, method, endpoint, data=None, expect_status=200):
    global PASS, FAIL
    if method == "GET":
        resp = client.get(endpoint)
    elif method == "POST":
        resp = client.post(endpoint, json=data)
    else:
        resp = client.get(endpoint)

    status = resp.status_code
    result = resp.json()

    if status == expect_status:
        PASS += 1
        print(f"  [PASS] {name} (status {status})")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}: expected {expect_status}, got {status}")
    return result


SAMPLE_DRILLING = "Drilling report for well Cantarell-X1. Total depth reached 12500 feet in the Cretaceous formation. Drilling fluid weight was 12.5 ppg. Lost circulation zones encountered at depth. Wellhead pressure recorded at 2500 psi."
SAMPLE_COMPLETION = "Completion report for well Pemex-A1 at depth 9800 ft in Ku-Nu formation. Perforated interval: 9500-9800. Equipment used: perforating gun. Initial production: 850.5 bbl/day oil. Successful completion with excellent production rates."


def main():
    global PASS, FAIL
    print("=" * 60)
    print("  NLP WELL REPORT ANALYZER - API TESTS")
    print("=" * 60)

    print("\n--- Health Check ---")
    test("GET /api/health", "GET", "/api/health")

    print("\n--- Models Info ---")
    r = test("GET /api/models", "GET", "/api/models")
    if r:
        print(f"    Models: {list(r.keys())}")

    print("\n--- Full Analysis ---")
    r = test("POST /api/analyze (drilling)", "POST", "/api/analyze", {"text": SAMPLE_DRILLING})
    if r:
        print(f"    Classification: {r.get('classification', {}).get('type', 'N/A')}")
        print(f"    Sentiment: {r.get('sentiment', {}).get('label', 'N/A')}")
        print(f"    Entities: well={r.get('entities', {}).get('well_name', 'N/A')}")
        print(f"    Keywords: {r.get('keywords', [])[:5]}")

    r = test("POST /api/analyze (completion)", "POST", "/api/analyze", {"text": SAMPLE_COMPLETION})
    if r:
        print(f"    Classification: {r.get('classification', {}).get('type', 'N/A')}")
        print(f"    Sentiment: {r.get('sentiment', {}).get('label', 'N/A')}")

    print("\n--- Classification ---")
    r = test("POST /api/classify", "POST", "/api/classify", {"text": SAMPLE_DRILLING})
    if r:
        print(f"    Result: {r.get('classification', 'N/A')}")
        print(f"    Probabilities: {r.get('probabilities', {})}")

    print("\n--- Entity Extraction ---")
    r = test("POST /api/extract", "POST", "/api/extract", {"text": SAMPLE_DRILLING})
    if r:
        print(f"    Well: {r.get('entities', {}).get('well_name', 'N/A')}")
        print(f"    Formation: {r.get('entities', {}).get('formation', 'N/A')}")
        print(f"    Depths: {r.get('entities', {}).get('depths', [])}")
        print(f"    Equipment: {r.get('entities', {}).get('equipment', [])}")

    print("\n--- Error Handling ---")
    test("POST /api/analyze (empty)", "POST", "/api/analyze", {"text": ""}, expect_status=400)
    test("POST /api/classify (no text)", "POST", "/api/classify", {}, expect_status=422)

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"  RESULTS: {PASS}/{total} passed, {FAIL} failed")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
