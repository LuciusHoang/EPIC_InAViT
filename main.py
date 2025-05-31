import argparse

from train import main as train_main
from evaluate import main as evaluate_main

def run():
    parser = argparse.ArgumentParser(description="EgoDex Classification Pipeline")

    parser.add_argument('--mode', type=str, choices=['train', 'evaluate'], default='train',
                        help='Operation mode: train or evaluate.')
    parser.add_argument('--model', type=str, choices=['mlp_bc', 'cnn_lstm', 'transformer'], default='mlp_bc',
                        help='Model architecture to use.')
    parser.add_argument('--checkpoint', type=str, default='logs/checkpoints/best_model.pth',
                        help='Path to model checkpoint (for evaluation).')

    args = parser.parse_args()

    if args.mode == 'train':
        train_main(model_name=args.model)
    elif args.mode == 'evaluate':
        evaluate_main(model_name=args.model, checkpoint_path=args.checkpoint)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    run()
