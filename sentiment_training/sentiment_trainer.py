import os
import time
import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from datasets import load_metric

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler


class SentimentTrainer():
    def __init__(self, args):
        self.args = args

        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Load the train and evaluation datasets
        dataset = load_dataset(args.dataset_name)
        train_dataset = dataset["train"].shuffle(seed=args.seed).select([i for i in list(range(args.num_train))])
        test_dataset = dataset["test"].shuffle(seed=args.seed).select([i for i in list(range(args.num_test))])

        # Load the pre-trained tokenizer model, tokenize the datasets, and pre-pare for training with PyTorch
        self.tokenizer = AutoTokenizer.from_pretrained(args.language_model)

        tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_train = tokenized_train.remove_columns(["text"])
        tokenized_train = tokenized_train.rename_column("label", "labels")
        tokenized_train.set_format("torch")

        tokenized_test = test_dataset.map(self.preprocess_function, batched=True)
        tokenized_test = tokenized_test.remove_columns(["text"])
        tokenized_test = tokenized_test.rename_column("label", "labels")
        tokenized_test.set_format("torch")

        # Initialize the PyTorch dataloaders for training
        self.train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=self.args.batch_size)
        self.eval_dataloader = DataLoader(tokenized_test, batch_size=self.args.batch_size)

        # Load the pre-trained language model for sentiment classification
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.language_model, num_labels=3)
        self.model.to(self.device)

        # Initialize the optimizer and scheduler used to fine-tune the sentiment-classification model
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate)
        self.num_training_steps = self.args.epochs * len(self.train_dataloader)

        self.lr_scheduler = get_scheduler(name="linear", optimizer=self.optimizer, num_warmup_steps=0,
                                     num_training_steps=self.num_training_steps)

        self.accuracy_metric = load_metric("accuracy")

        self.statistic_template = {"Epoch": 0,
                                   "Average Training Epoch Loss": 0,
                                   "Average Validation Epoch Loss": 0,
                                   "Average Validation Epoch Accuracy": 0}

        self.train_statistics = []

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)

    def train(self):
        train_init_time = time.time()

        for epoch in range(self.args.epochs):

            epoch_statistic = self.statistic_template
            epoch_statistic["Epoch"] = epoch + 1

            print('\n======== Epoch {:}/{:} ========'.format(epoch + 1, self.args.epochs))
            print('\nTraining...')
            train_epoch_progress_bar = tqdm(range(len(self.train_dataloader)))
            total_epoch_train_loss = 0
            self.model.train()
            for step, batch in enumerate(self.train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                loss = outputs.loss
                total_epoch_train_loss += loss.item()
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                train_epoch_progress_bar.update(1)
                time.sleep(0.1)
            train_epoch_progress_bar.close()

            avg_epoch_train_loss = total_epoch_train_loss/len(self.train_dataloader)
            epoch_statistic["Average Training Epoch Loss"] = avg_epoch_train_loss
            print("\nTraining epoch finished! Time elapsed: {}".format(time.time() - train_init_time))
            print("Average epoch training loss: {0:.4f}".format(avg_epoch_train_loss))



            print("\nValidating current model performance...")
            eval_epoch_progress_bar = tqdm(range(len(self.eval_dataloader)))

            total_epoch_val_loss = 0
            self.model.eval()
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.no_grad():
                    outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits

                total_epoch_val_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                self.accuracy_metric.add_batch(predictions=predictions, references=batch["labels"])
                eval_epoch_progress_bar.update(1)
            eval_epoch_progress_bar.close()

            avg_epoch_accuracy = self.accuracy_metric.compute()
            avg_epoch_val_loss = total_epoch_val_loss/len(self.eval_dataloader)
            epoch_statistic["Average Validation Epoch Loss"] = avg_epoch_val_loss
            epoch_statistic["Average Epoch Accuracy"] = avg_epoch_accuracy
            print("\nValidation epoch finished! Time elapsed: {0:.4f}".format(time.time() - train_init_time))
            print("Average epoch validation loss: {0:.4f}".format(avg_epoch_val_loss))
            print("Average epoch validation accuracy: {0:.4f}".format(avg_epoch_val_loss))

            self.train_statistics.append(epoch_statistic)

        print("\nTraining Finished! Total Training Time: {}".format(time.time() - train_init_time))


    def save_model(self):
        save_dir = "./{}/{}".format(self.args.output_dir, self.args.model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print("Saving model to {}".format(save_dir))
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
