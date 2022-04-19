import os
import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras import optimizers
from matplotlib.ticker import MaxNLocator
from tensorflow.keras import regularizers, initializers
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from datetime import timedelta 
from datetime import datetime
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix
# Load the train and evaluation datasets

class BuySellTrainer():
  def __init__(self, args):
    self.args = args
    self.svm = svm
    self.LogisticRegression = LogisticRegression
    np.random.seed(self.args.seed)

    # Load the train and evaluation datasets
    db_names = ["Date", "time", "tweet", "sv", "score"] 
    stock_dataset = pd.read_csv(args.stock_dataset_name+"/stock_data.csv")
    train_dataset = pd.read_csv(args.sentimental_dataset_name+"/train_data.csv",
                                names=db_names)
    test_dataset = pd.read_csv(args.sentimental_dataset_name+"/test_data.csv",
                                names=db_names)
    stock_dataset.set_index(pd.DatetimeIndex(stock_dataset['Date']), inplace=True)
    train_dataset.set_index(pd.DatetimeIndex(train_dataset['Date']), inplace=True)
    test_dataset.set_index(pd.DatetimeIndex(test_dataset['Date']), inplace=True)
    n_train_dataset = pd.merge(stock_dataset, train_dataset, how='inner', 
                                   left_index=True, right_index=True)
    n_test_dataset = pd.merge(stock_dataset, test_dataset, how='outer', 
                                   left_index=True, right_index=True)
    
    n_train = len(train_dataset.values)
    n_test = len(test_dataset.values)
    n_stock = len(stock_dataset.values)
    close_time = datetime.strptime("16:00:00", '%H:%M:%S')
    open_time = datetime.strptime("10:00:00", '%H:%M:%S')
    pos_r, neg_r, neu_r, n_tw_lst, score_lst, signal_lst = [],[],[],[],[],[]
    dates = []
    for i in range(n_train):
      today = train_dataset["Date"].index[i]
      if not today in dates:
        dates.append(today)
        hour = datetime.strptime(train_dataset["time"].values[i], '%H:%M:%S')
        pos = 0
        neg = 0
        neu = 0
        score = 0
        n_tw = 0
        for j in range(n_train):
          tw_hour = datetime.strptime(train_dataset["time"][j], '%H:%M:%S')
          #Here we need to be careful, a stock market day it is consider from the close
          #the previous day till the next close. 
          if hour > close_time: 
            if today == train_dataset["Date"].index[j]:
              if tw_hour > close_time:
                sv_tw = int(train_dataset["sv"].values[j])
                if sv_tw == 0:
                  neg += 1
                elif sv_tw == 1:
                  neu += 1
                else:
                  pos += 1
                score += train_dataset["score"].values[j]
                n_tw += 1
            elif today + timedelta(days=1) == train_dataset["Date"].index[j]:
              if tw_hour < close_time:
                sv_tw = int(train_dataset["sv"].values[j])
                if sv_tw == 0:
                  neg += 1
                elif sv_tw == 1:
                  neu += 1
                else:
                  pos += 1
                score += train_dataset["score"].values[j]
                n_tw += 1
            else:
              pass

          elif hour < close_time:
            if today == train_dataset["Date"].index[j]:
              if tw_hour < close_time:
                sv_tw = int(train_dataset["sv"].values[j])
                if sv_tw == 0:
                  neg += 1
                elif sv_tw == 1:
                  neu += 1
                else:
                  pos += 1
                score += train_dataset["score"].values[j]
                n_tw += 1
            elif today - timedelta(days=1) == train_dataset["Date"].index[j]:
              if tw_hour > close_time:
                sv_tw = int(train_dataset["sv"].values[j])
                if sv_tw == 0:
                  neg += 1
                elif sv_tw == 1:
                  neu += 1
                else:
                  pos += 1
                score += train_dataset["score"].values[j]
                n_tw += 1
            else:
              pass

        pos_r.append(pos/n_tw)
        neg_r.append(neg/n_tw)
        neu_r.append(neu/n_tw)
        score_lst.append(score)
        n_tw_lst.append(n_tw)

        try:
          if hour > close_time:
            tomorrow = today + timedelta(days=1)
            dif = stock_dataset["Open"][tomorrow] - stock_dataset["Close/Last"][tomorrow]
            if dif > 0:
                signal_lst.append(1) #BUY
            else:
                signal_lst.append(0) #SELL
          elif hour < close_time:
            dif = stock_dataset["Open"][today] - stock_dataset["Close/Last"][today]
            if dif > 0:
              signal_lst.append(1) #BUY
            else:
              signal_lst.append(0) #SELL
        except:
          signal_lst.append(1)
    
    self.x_train = np.array([pos_r, neg_r, neu_r, n_tw_lst, score_lst]).T
    self.x_train[:,3] = self.x_train[:,3]/np.amax(self.x_train[:,3])
    self.x_train[:,4] = self.x_train[:,4]/np.amax(self.x_train[:,4])
    self.y_train = np.array(signal_lst)

    pos_r, neg_r, neu_r, n_tw_lst, score_lst, signal_lst = [],[],[],[],[],[]
    dates = []
    for i in range(n_test):
      today = test_dataset["Date"].index[i]
      if not today in dates:
        dates.append(today)
        hour = datetime.strptime(test_dataset["time"].values[i], '%H:%M:%S')
        pos = 0
        neg = 0
        neu = 0
        score = 0
        n_tw = 0
        for j in range(n_test):
          tw_hour = datetime.strptime(test_dataset["time"][j], '%H:%M:%S')
          #Here we need to be careful, a stock market day it is consider from the close
          #the previous day till the next close. 
          if hour > close_time: 
            if today == test_dataset["Date"].index[j]:
              if tw_hour > close_time:
                sv_tw = int(test_dataset["sv"].values[j])
                if sv_tw == 0:
                  neg += 1
                elif sv_tw == 1:
                  neu += 1
                else:
                  pos += 1
                score += test_dataset["score"].values[j]
                n_tw += 1
            elif today + timedelta(days=1) == test_dataset["Date"].index[j]:
              if tw_hour < close_time:
                sv_tw = int(test_dataset["sv"].values[j])
                if sv_tw == 0:
                  neg += 1
                elif sv_tw == 1:
                  neu += 1
                else:
                  pos += 1
                score += test_dataset["score"].values[j]
                n_tw += 1
            else:
              pass

          elif hour < close_time:
            if today == test_dataset["Date"].index[j]:
              if tw_hour < close_time:
                sv_tw = int(test_dataset["sv"].values[j])
                if sv_tw == 0:
                  neg += 1
                elif sv_tw == 1:
                  neu += 1
                else:
                  pos += 1
                score += test_dataset["score"].values[j]
                n_tw += 1
            elif today - timedelta(days=1) == test_dataset["Date"].index[j]:
              if tw_hour > close_time:
                sv_tw = int(test_dataset["sv"].values[j])
                if sv_tw == 0:
                  neg += 1
                elif sv_tw == 1:
                  neu += 1
                else:
                  pos += 1
                score += test_dataset["score"].values[j]
                n_tw += 1
            else:
              pass

        pos_r.append(pos/n_tw)
        neg_r.append(neg/n_tw)
        neu_r.append(neu/n_tw)
        score_lst.append(score)
        n_tw_lst.append(n_tw)

        try:
          if hour > close_time:
            tomorrow = today + timedelta(days=1)
            dif = stock_dataset["Open"][tomorrow] - stock_dataset["Close/Last"][tomorrow]
            if dif > 0:
                signal_lst.append(1) #BUY
            else:
                signal_lst.append(0) #SELL
          elif hour < close_time:
            dif = stock_dataset["Open"][today] - stock_dataset["Close/Last"][today]
            if dif > 0:
              signal_lst.append(1) #BUY
            else:
              signal_lst.append(0) #SELL
        except:
          signal_lst.append(1)


    self.x_test = np.array([pos_r, neg_r, neu_r, n_tw_lst, score_lst]).T
    self.x_test[:,3] = self.x_test[:,3]/np.amax(self.x_test[:,3])
    self.x_test[:,4] = self.x_test[:,4]/np.amax(self.x_test[:,4])
    self.y_test = np.array(signal_lst)

    self.x_val = self.x_train[0:50]
    self.y_val = self.y_train[0:50]

    self.statistic_template = {"NN model":[],
							       "Training Epoch Loss": [],
		                           "Training Epoch Accuracy": [],
		                           "Validation Epoch Loss": [],
		                           "Validation Epoch Accuracy": []
		                           }
    self.results = {"No regularizer": [],
						"L2": [],
						"BN": [],
						"Dropout": []
						}

    self.train_statistics = []
    # import ipdb; ipdb.set_trace(context=15)
  def train_nn(self):
    epoch_statistic = self.statistic_template
    res = self.results

    train_init_time = time.time()

    #No regularizer
    model = Sequential()
    model.add(Dense(units=self.args.nn_units, activation='relu', 
    input_shape=[self.x_train.shape[1]]))
    model.add(Dense(units=self.args.nn_units, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=self.args.learning_rate) 
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
    metrics=['accuracy'])
    model.summary()
    history = model.fit(self.x_train, self.y_train, epochs=self.args.epochs,
    batch_size=self.args.batch_size, validation_data=(self.x_val,self.y_val))
    results = model.evaluate(self.x_test, self.y_test)

    epoch_statistic['NN model'].append("No regularizer")
    epoch_statistic["Training Epoch Loss"].append(history.history['loss'])
    epoch_statistic["Validation Epoch Loss"].append(history.history['val_loss'])
    epoch_statistic["Training Epoch Accuracy"].append(history.history['accuracy'])
    epoch_statistic["Validation Epoch Accuracy"].append(history.history['val_accuracy'])
    res["No regularizer"].append(results)
    self.save_model(model, "No_regularizer")

    ##With L2 regularizer
    model = Sequential()
    model.add(Dense(units=self.args.nn_units, activation='relu', 
    input_shape=[self.x_train.shape[1]],
    kernel_regularizer=regularizers.l2(1e-3),
    bias_regularizer=regularizers.l2(1e-2)))
    model.add(Dense(units=self.args.nn_units, activation='relu',
    kernel_regularizer=regularizers.l2(1e-3),
    bias_regularizer=regularizers.l2(1e-2)))
    model.add(Dense(units=1, activation='sigmoid',
    kernel_regularizer=regularizers.l2(1e-3),
    bias_regularizer=regularizers.l2(1e-2)))
    optimizer = optimizers.RMSprop(lr=self.args.learning_rate) 
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
    metrics=['accuracy'])
    model.summary()
    history = model.fit(self.x_train, self.y_train, epochs=self.args.epochs, 
    batch_size=self.args.batch_size,
    validation_data=(self.x_val,self.y_val))
    results = model.evaluate(self.x_test, self.y_test)

    epoch_statistic['NN model'].append("L2 regularizer")
    epoch_statistic["Training Epoch Loss"].append(history.history['loss'])
    epoch_statistic["Validation Epoch Loss"].append(history.history['val_loss'])
    epoch_statistic["Training Epoch Accuracy"].append(history.history['accuracy'])
    epoch_statistic["Validation Epoch Accuracy"].append(history.history['val_accuracy'])
    res["L2"].append(results)
    self.save_model(model, "L2_regularizer")


    ##With BatchNormalization (Almost mandatory)
    model = Sequential()
    model.add(Dense(units=self.args.nn_units, activation='relu',
                    input_shape=[self.x_train.shape[1]]))
    model.add(BatchNormalization())
    model.add(Dense(units=self.args.nn_units, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=1, activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=self.args.learning_rate) 
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
    metrics=['accuracy'])
    model.summary()
    history = model.fit(self.x_train, self.y_train, epochs=self.args.epochs, batch_size=self.args.batch_size,
    validation_data=(self.x_val,self.y_val))
    results = model.evaluate(self.x_test, self.y_test)

    epoch_statistic['NN model'].append("Batch Normalization")
    epoch_statistic["Training Epoch Loss"].append(history.history['loss'])
    epoch_statistic["Validation Epoch Loss"].append(history.history['val_loss'])
    epoch_statistic["Training Epoch Accuracy"].append(history.history['accuracy'])
    epoch_statistic["Validation Epoch Accuracy"].append(history.history['val_accuracy'])
    res["BN"].append(results)
    self.save_model(model, "Batch_Normalization")



    ##With DROPOUT
    model = Sequential()
    model.add(Dense(units=self.args.nn_units, activation='relu', 
                    input_shape=[self.x_train.shape[1]]))
    model.add(Dropout(0.5))
    model.add(Dense(units=self.args.nn_units, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=self.args.learning_rate) 
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
    metrics=['accuracy'])
    model.summary()
    history = model.fit(self.x_train, self.y_train, epochs=self.args.epochs,
                        batch_size=self.args.batch_size,
    validation_data=(self.x_val,self.y_val))

    results = model.evaluate(self.x_test, self.y_test)

    epoch_statistic['NN model'].append("Batch Normalization")
    epoch_statistic["Training Epoch Loss"].append(history.history['loss'])
    epoch_statistic["Validation Epoch Loss"].append(history.history['val_loss'])
    epoch_statistic["Training Epoch Accuracy"].append(history.history['accuracy'])
    epoch_statistic["Validation Epoch Accuracy"].append(history.history['val_accuracy'])
    res["Dropout"].append(results)
    self.save_model(model, "Dropout")

    print("\nTraining Finished! Total Training Time: {}".format(time.time() - train_init_time))

  def plot_nn_train(self):
    st = self.statistic_template
    train_loss = st["Training Epoch Loss"] 
    val_loss = st["Validation Epoch Loss"]
    train_acc = st["Training Epoch Accuracy"]
    val_acc = st["Validation Epoch Accuracy"]

    fnt = 17
    fig_t=plt.figure(figsize=(8,5))
    plt.plot(train_loss[0],label="Train_NR", color="blue", alpha=0.5)
    plt.plot(train_loss[1],label="Train_L2", color="green", alpha=0.5)
    plt.plot(train_loss[2],label="Train_BN", color="red", alpha=0.5)
    plt.plot(train_loss[2],label="Train_Dropout", color="orange", alpha=0.5)
    plt.plot(val_loss[0],label="Val_NR", color="blue")
    plt.plot(val_loss[1],label="Val_L2", color="green")
    plt.plot(val_loss[2],label="Val_BN", color="red")
    plt.plot(val_loss[3],label="Val_Dropout", color="orange")
    plt.xlabel(r"Epochs",fontsize=fnt+1)                                         
    plt.ylabel(r"Loss",fontsize=fnt+1)
    plt.title(r"Loss as a function of epochs", fontsize=fnt+2)
    plt.xticks(fontsize=fnt-2) 
    plt.yticks(fontsize=fnt-2)   
    plt.legend(bbox_to_anchor=(0.1,0.85),ncol=2,fontsize=fnt-5,framealpha=1,loc=6) 
    plt.tight_layout()
    plt.grid()
    fig_t.savefig("Loss_NN.pdf")

    ##PLOTEOS
    fnt = 17
    fig_t=plt.figure(figsize=(8,5))
    plt.plot(train_acc[0],label="Train_NR", color="blue", alpha=0.5)
    plt.plot(train_acc[1],label="Train_L2", color="green", alpha=0.5)
    plt.plot(train_acc[2],label="Train_BN", color="red", alpha=0.5)
    plt.plot(train_acc[3],label="Train_Dropout", color="orange", alpha=0.5)
    plt.plot(val_acc[0],label="Val_NR", color="blue")
    plt.plot(val_acc[1],label="Val_L2", color="green")
    plt.plot(val_acc[2],label="Val_BN", color="red")
    plt.plot(val_acc[3],label="Val_Dropout", color="orange")
    plt.xlabel(r"Epochs",fontsize=fnt+1)                                         
    plt.ylabel(r"Accuracy",fontsize=fnt+1)
    plt.title(r"Accuracy as a function of epochs", fontsize=fnt+2)
    plt.xticks(fontsize=fnt-2) 
    plt.yticks(fontsize=fnt-2)   
    plt.legend(bbox_to_anchor=(0.1,0.2),ncol=2,fontsize=fnt-5,framealpha=1,loc=6) 
    plt.tight_layout()
    plt.grid()
    fig_t.savefig("Acc_NN.pdf")
    plt.show()

  def svm_train(self):
    # Initialize SVM classifier
    clf = self.svm.SVC(kernel='linear')

    # Fit data
    clf = clf.fit(self.x_train, self.y_train)
    predictions = clf.predict(self.x_test)
    # Generate confusion matrix
    matrix = plot_confusion_matrix(clf, self.x_test, self.y_test,
                                    cmap=plt.cm.Blues,
                                    normalize='true')
    plt.title('Confusion matrix for our classifier')
    plt.show(matrix)
    plt.show()

  def LogisticRegression_train(self):
    model = self.LogisticRegression()
    model.fit(self.x_train, self.y_train)
    y_pred = model.predict(self.x_test)
    print("Accuracy:", metrics.accuracy_score(self.y_test, y_pred))
    print("Precision:", metrics.precision_score(self.y_test, y_pred))
    print("Recall:", metrics.recall_score(self.y_test, y_pred))
    # Generate confusion matrix
    matrix = plot_confusion_matrix(model, self.x_test, self.y_test,
                                    cmap=plt.cm.Blues,
                                    normalize='true')
    plt.title('Confusion matrix for Logistic')
    plt.show(matrix)
    plt.show()

  def save_model(self, model, regularizer):
    save_dir = "./{}/{}".format(self.args.output_dir, self.args.model_name+regularizer)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Saving model to {}".format(save_dir))
    model.save_weights(save_dir, overwrite=True, save_format=None, options=None)
