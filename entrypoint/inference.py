import argparse

import yaml
from dotenv import load_dotenv

from src.pipelines.inference_pipeline import run


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run inference with a trained VQA Chest model")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained .pt checkpoint")
    parser.add_argument("--image", help="Path to an image for a single prediction")
    parser.add_argument("--question", help="Question to ask for a single prediction")
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split for batch prediction when --image/--question are omitted",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run(
        config,
        args.checkpoint,
        image=args.image,
        question=args.question,
        split=args.split,
    )


if __name__ == "__main__":
    main()
