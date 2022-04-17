
from ECE6254FinalProject.sentiment_training.sentiment_trainer import SentimentTrainer
from ECE6254FinalProject.sentiment_training.sentiment_train_options import SentimentTrainOptions

if __name__ == "__main__":

    options = SentimentTrainOptions()
    trainer = SentimentTrainer(options.args)

    trainer.train()
    trainer.save_model()


