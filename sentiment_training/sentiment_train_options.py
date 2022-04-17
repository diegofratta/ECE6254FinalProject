import os
import argparse

class SentimentTrainOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Argument Options for Training Sentiment Analysis")

        self.parser.add_argument("--seed",
                                 type=int,
                                 help="Random seed to be used for reproducibility",
                                 default=42)

        """
        Dataset parameters.
        """
        self.parser.add_argument("--dataset_name",
                                 type=str,
                                 help="Directory to the training dataset to be used using the datasets library.",
                                 default="dataset/tesla_stock_tweets")

        self.parser.add_argument("--num_train",
                                 type=int,
                                 help="Number of training samples to use during training.",
                                 default=8)

        self.parser.add_argument("--num_test",
                                 type=int,
                                 help="Number of testing samples to use during training.",
                                 default=8)

        """
        transformers library parameters/ training parameters
        """
        self.parser.add_argument("--language_model",
                                 type=str,
                                 help="Name of the tokenizer model used with the transformers library.",
                                 default="distilbert-base-uncased")

        self.parser.add_argument("--output_dir",
                                 type=str,
                                 help="Name of the model output directory",
                                 default="training_models")

        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="Name of the model output directory",
                                 default="tesla_stock_tweet_sentiment_model")

        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="Learning rate for training the sentiment analysis model.",
                                 default=2e-5)

        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="Batch size during training.",
                                 default=2)

        self.parser.add_argument("--epochs",
                                 type=int,
                                 help="Number of training epochs.",
                                 default=2)

        # Arguments added at the end in order to use this Object with a Jupyter Notebook
        self.parser.add_argument('strings',
                                 metavar='STRING',
                                 nargs='*',
                                 help='String for searching',)

        self.parser.add_argument('-f',
                                 '--file',
                                 help='Path for input file. First line should contain number of lines to search in')

        self.args = self.parser.parse_args()
