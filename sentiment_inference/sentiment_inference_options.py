import os
import argparse

class SentimentInferenceOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Argument Options for initializing a sentiment analysis model for inference")

        self.parser.add_argument("--load_model_directory",
                                 type=str,
                                 help="Name of the tokenizer model used with the transformers library.",
                                 default="./training_models/tesla_stock_tweet_sentiment_model/")

        # Arguments added at the end in order to use this Object with a Jupyter Notebook
        self.parser.add_argument('strings',
                                 metavar='STRING',
                                 nargs='*',
                                 help='String for searching',)

        self.parser.add_argument('-f',
                                 '--file',
                                 help='Path for input file. First line should contain number of lines to search in')

        self.args = self.parser.parse_args()
