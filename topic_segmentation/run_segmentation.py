# topic_segmentation/run_segmentation.py

import subprocess
import sys

def run_segmentation(input_json: str, algorithm: str):
    cmd = [
        "python",
        "topic_segmentation_core.py",
        input_json,
        algorithm
    ]

    print(f"▶ Running topic segmentation [{algorithm}]")
    subprocess.run(cmd, check=True)
    print("✅ Topic segmentation completed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_segmentation.py <pipeline_output.json> [algorithm]")
        sys.exit(1)

    input_json = sys.argv[1]
    algorithm = sys.argv[2] if len(sys.argv) > 2 else "baseline_similarity"

    run_segmentation(input_json, algorithm)
