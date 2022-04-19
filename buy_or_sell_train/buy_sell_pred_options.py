import os
import argparse

class BuySellOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Argument Options for Training the buy or sell predictor")

        self.parser.add_argument("--seed",
            type=int,
            help="Random seed to be used for reproducibility",
            default=42)

        """
        Dataset parameters.
        """
        self.parser.add_argument("--stock_dataset_name",
             type=str,
             help="Directory to the stock training dataset to be used using the datasets library.",
             default="dataset/tesla_stock_data")

        self.parser.add_argument("--sentimental_dataset_name",
             type=str,
             help="Directory to the sentimental analysis training dataset to be used using the datasets library.",
             default="dataset/tesla_sentimental_data")

        self.parser.add_argument("--num_train",
             type=int,
             help="Number of training samples to use during training.",
             default=8)

        self.parser.add_argument("--num_test",
             type=int,
             help="Number of testing samples to use during testing.",
             default=8)
        """
        transformers library parameters/ training parameters
        """
        self.parser.add_argument("--output_dir",
             type=str,
             help="Name of the model output directory",
             default="training_models")

        self.parser.add_argument("--model_name",
             type=str,
             help="Name of the model output directory",
             default="tesla_stock_buy_or_sell_model")

        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="Learning rate for training the sentiment analysis model.",
                                 default=2e-5)

        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="Batch size during training.",
                                 default=4)

        self.parser.add_argument("--epochs",
                                 type=int,
                                 help="Number of training epochs.",
                                 default=100)

        self.parser.add_argument("--nn_units",
                                 type=int,
                                 help="Size of the output from the dense layer.",
                                 default=16)

        # Arguments added at the end in order to use this Object with a Jupyter Notebook
        self.parser.add_argument('strings',
                                 metavar='STRING',
                                 nargs='*',
                                 help='String for searching',)

        self.parser.add_argument('-f',
                                 '--file',
                                 help='Path for input file. First line should contain number of lines to search in')

        self.args = self.parser.parse_args()