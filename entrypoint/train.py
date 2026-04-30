import yaml
import argparse # Because we want to be able to specify the config file from the command line
from src.pipelines.training_pipeline import run_training # This is the main function that will run the training pipeline

parser = argparse.ArgumentParser() # This is the argument parser that will parse the command line arguments
parser.add_argument("--config", required=True) # We require a config file to be specified from the command line
args = parser.parse_args() # Parse the command line arguments and store them in the args variable

config = yaml.safe_load(open(args.config)) # Load the config file specified from the command line
run_training(config) # Run the training pipeline with the loaded config