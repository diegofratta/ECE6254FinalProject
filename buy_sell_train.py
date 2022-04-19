from ECE6254FinalProject.buy_or_sell_train.buy_or_sell_trainer import BuySellTrainer
from ECE6254FinalProject.buy_or_sell_train.buy_sell_pred_options import BuySellOptions

if __name__ == "__main__":
  options = BuySellOptions()
  trainer = BuySellTrainer(options.args)

  trainer.train_nn()
  trainer.plot_nn_train()
  trainer.svm_train()
  trainer.LogisticRegression_train()