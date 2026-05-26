import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Read the latest report next to this script so the file can move with the repo.
INPUT_FILE = PROJECT_ROOT / "results_nanogcg.json"

def main():
    if not INPUT_FILE.exists():
        print(f"File not found: {INPUT_FILE}")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    success_cases = []

    for result in data.get("results", []):
        # We only care about tests that successfully triggered the target marker
        if result.get("transfer_asr", 0.0) > 0:
            base_msg = result.get("base_message")
            for eval_item in result.get("evaluation", []):
                if eval_item.get("contains_target_marker") is True:
                    success_cases.append({
                        "base_message": base_msg,
                        "response": eval_item.get("response")
                    })
    
    print(f"Found {len(success_cases)} successful cases:\n")
    for i, case in enumerate(success_cases, 1):
        print(f"=== Case {i} ===")
        print(f"Base Message: {case['base_message']}")
        print(f"Response    :\n{case['response'].strip()}")
        print("-" * 50 + "\n")
    print(",".join([case['base_message'] for case in success_cases]))

if __name__ == "__main__":
    main()
