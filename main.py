# main.py

import argparse
import sys
from generate_test_labels import generate_test_labels_and_ids
from precompute_features import run_feature_extraction
from evaluate import evaluate
from ensemble import run_ensemble

def run():
    parser = argparse.ArgumentParser(description="EgoDex Full Pipeline")
    parser.add_argument('--run_all', action='store_true', help='Run the full pipeline: labels → features → evaluate → ensemble')
    parser.add_argument('--evaluate_only', choices=['egom2p', 'egopack'], help='Evaluate only one model')
    args = parser.parse_args()

    if args.run_all:
        print("\U0001f527 Step 1: Generate labels...")
        generate_test_labels_and_ids()

        for model in ['egom2p', 'egopack']:
            print(f"\U0001f680 Step 2: Precompute features for {model}...")
            run_feature_extraction(embedding_type=model)

            print(f"\U0001f9ea Step 3: Evaluate {model}...")
            evaluate(embedding_type=model)

        print("⚖️ Step 4: Running ensemble...")
        run_ensemble()

    elif args.evaluate_only:
        print(f"\U0001f9ea Evaluating {args.evaluate_only} only...")
        evaluate(embedding_type=args.evaluate_only)

    else:
        print("❌ No mode specified. Use --run_all or --evaluate_only.")
        sys.exit(1)

if __name__ == "__main__":
    run()
