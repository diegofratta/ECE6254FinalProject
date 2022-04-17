
import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import pipeline

class SentimentInferenceModel():

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.load_model_directory)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.load_model_directory)

        self.sentiment_classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)

    def get_sentiment(self, tweet_text):
        output = self.sentiment_classifier(tweet_text)
        print(output)
