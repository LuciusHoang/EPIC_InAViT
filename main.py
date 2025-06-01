import argparse
import sys

from train import train
from evaluate import evaluate
import config

def run():
    parser = argparse.ArgumentParser(description="EgoDex Classification Pipeline")

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate'],
        default='train',
        help='Operation mode: train or evaluate.'
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['mlp_bc', 'cnn_lstm', 'transformer'],
        default='mlp_bc',
        help='Model architecture to use.'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (for evaluation).'
    )

    args = parser.parse_args()

    # Update config with selected model
    config.model_name = args.model

    if args.mode == 'train':
        train()  # <-- remove config argument
    elif args.mode == 'evaluate':
        if args.checkpoint is None:
            print("Error: Checkpoint path is required for evaluation mode.")
            sys.exit(1)
        evaluate(checkpoint_path=args.checkpoint)  # <-- remove config argument
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    run()
