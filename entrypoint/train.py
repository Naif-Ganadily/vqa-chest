import argparse
import yaml
from dotenv import load_dotenv
from src.pipelines.training_pipeline import run


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Train VQA Chest model")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run(config)


if __name__ == "__main__":
    main()
