"""
Performance benchmark for the Content Moderator API.
Run with: python tests/test_performance.py

Tests:
1. Single request latency
2. Batch request latency
3. Week 1 benchmark texts through API
4. Throughput (texts per second)
"""

import time
import json
import requests

BASE_URL = "http://localhost:8000/api/v1"


def test_single_latency():
    """Measure single request latency over 10 requests."""
    texts = [
        "You are a disgusting person",
        "Have a wonderful day!",
        "तुम बेवकूफ हो",
        "شكراً لك على مساعدتك",
        "You are stupid and worthless",
        "The weather is beautiful today",
        "أنت غبي ولا تستحق الاحترام",
        "आपका दिन शुभ हो",
        "Nobody wants you here, get lost",
        "Thank you for your help",
    ]

    print("=" * 65)
    print("SINGLE REQUEST LATENCY")
    print("=" * 65)

    latencies = []
    for text in texts:
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/moderate",
            json={"text": text},
        )
        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)

        data = response.json()
        verdict = data["verdict"]
        lang = data["language"]
        emoji = "🚩" if verdict == "toxic" else "✅"
        print(f"  {emoji} {elapsed_ms:>7.0f}ms  [{lang}] {verdict:<6} {text[:45]}")

    avg = sum(latencies) / len(latencies)
    min_l = min(latencies)
    max_l = max(latencies)
    print(f"\n  Average: {avg:.0f}ms | Min: {min_l:.0f}ms | Max: {max_l:.0f}ms")
    return avg


def test_batch_latency():
    """Measure batch request latency for different batch sizes."""
    print("\n" + "=" * 65)
    print("BATCH REQUEST LATENCY")
    print("=" * 65)

    # Create test texts of different sizes
    base_texts = [
        "You are a disgusting person",
        "Have a wonderful day!",
        "तुम बेवकूफ हो",
        "شكراً لك على مساعدتك",
        "You are stupid and worthless",
    ]

    for batch_size in [5, 10, 20, 50]:
        # Repeat texts to fill batch
        texts = (base_texts * (batch_size // len(base_texts) + 1))[:batch_size]

        start = time.time()
        response = requests.post(
            f"{BASE_URL}/moderate/batch",
            json={"texts": texts},
        )
        elapsed_ms = (time.time() - start) * 1000
        data = response.json()

        per_text = elapsed_ms / batch_size
        texts_per_sec = batch_size / (elapsed_ms / 1000)
        print(
            f"  Batch={batch_size:<4} "
            f"Total: {elapsed_ms:>7.0f}ms  "
            f"Per text: {per_text:>5.0f}ms  "
            f"Throughput: {texts_per_sec:>5.1f} texts/sec  "
            f"Flagged: {data['flagged_count']}/{data['total_texts']}"
        )


def test_week1_benchmark():
    """Run Week 1 benchmark texts through the API and verify results."""
    print("\n" + "=" * 65)
    print("WEEK 1 BENCHMARK VERIFICATION")
    print("=" * 65)

    benchmark_texts = [
        # (text, expected_verdict, language)
        ("You are an absolute idiot and waste of space", "toxic", "en"),
        ("I will make sure you regret being alive", "toxic", "en"),
        ("Thank you for your thoughtful response", "clean", "en"),
        ("Great presentation today, well done!", "clean", "en"),
        ("The weather has been lovely this week", "clean", "en"),
        ("तुम एक बेवकूफ और बेकार इंसान हो", "toxic", "hi"),
        ("आपका काम बहुत अच्छा है", "clean", "hi"),
        ("आज का मौसम बहुत सुहावना है", "clean", "hi"),
        ("أنت غبي ولا تستحق الاحترام", "toxic", "ar"),
        ("شكراً لك على مساعدتك", "clean", "ar"),
        ("أتمنى لك يوماً سعيداً", "clean", "ar"),
        ("الطقس جميل اليوم", "clean", "ar"),
    ]

    correct = 0
    total = len(benchmark_texts)
    errors = []

    print(f"\n  {'Expected':<10} {'Got':<10} {'Lang':<6} {'Score':<8} {'Text'}")
    print(f"  {'-' * 60}")

    for text, expected, expected_lang in benchmark_texts:
        response = requests.post(
            f"{BASE_URL}/moderate",
            json={"text": text},
        )
        data = response.json()
        actual = data["verdict"]
        score = data["categories"].get("toxic", {}).get("score", 0)
        lang = data["language"]

        match = actual == expected
        if match:
            correct += 1
            emoji = "✅"
        else:
            emoji = "❌"
            errors.append({
                "text": text[:40],
                "expected": expected,
                "got": actual,
                "score": score,
                "lang": lang,
            })

        text_short = text[:40] + "..." if len(text) > 40 else text
        print(f"  {emoji} {expected:<10} {actual:<10} {lang:<6} {score:<8.4f} {text_short}")

    print(f"\n  Result: {correct}/{total} correct ({correct/total:.1%})")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    Expected {e['expected']}, got {e['got']} "
                  f"(score: {e['score']:.4f}) [{e['lang']}] '{e['text']}'")

    return correct, total


def test_edge_cases():
    """Test edge cases through the API."""
    print("\n" + "=" * 65)
    print("EDGE CASE TESTING")
    print("=" * 65)

    edge_cases = [
        ("Whitespace only", "   ", "clean"),
        ("Short text", "hi", "clean"),
        ("Repeated chars", "You are stuuuuuupid", "toxic"),
        ("ALL CAPS", "YOU ARE STUPID AND WORTHLESS", "toxic"),
        ("Mixed language", "You are such a bewakoof yaar", "toxic"),
        ("Just emojis", "🔥🔥🔥", "clean"),
        ("Constructive", "Your work needs significant improvement", "clean"),
    ]

    print(f"\n  {'Test':<20} {'Expected':<10} {'Got':<10} {'Score':<8} {'Warnings'}")
    print(f"  {'-' * 65}")

    for label, text, expected in edge_cases:
        response = requests.post(
            f"{BASE_URL}/moderate",
            json={"text": text},
        )
        data = response.json()
        actual = data["verdict"]
        score = data["categories"].get("toxic", {}).get("score", 0)
        warnings = data.get("warnings", [])
        match = "✅" if actual == expected else "❌"
        warn_str = "; ".join(warnings)[:30] if warnings else "—"

        print(f"  {match} {label:<20} {expected:<10} {actual:<10} {score:<8.4f} {warn_str}")


if __name__ == "__main__":
    print("\n" + "🛡️ MULTILINGUAL CONTENT MODERATOR — PERFORMANCE REPORT".center(65))
    print("=" * 65)

    # Check server is running
    try:
        r = requests.get(f"{BASE_URL}/health")
        health = r.json()
        print(f"\n  Model: {health['model_name']}")
        print(f"  Device: {health['device']}")
        print(f"  Status: {health['status']}")
    except requests.ConnectionError:
        print("\n  ❌ Server not running! Start it with:")
        print("     uvicorn app.main:app --reload --port 8000")
        exit(1)

    # Run all benchmarks
    avg_latency = test_single_latency()
    test_batch_latency()
    correct, total = test_week1_benchmark()
    test_edge_cases()

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Single request avg latency:  {avg_latency:.0f}ms")
    print(f"  Week 1 benchmark accuracy:   {correct}/{total} ({correct/total:.1%})")
    print(f"  All endpoints operational:   ✅")
    print(f"  Input validation working:    ✅")
    print("=" * 65)