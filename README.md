# ECE6254FinalProject
 Final Project for ECE 6254 at GaTech

### Dependencies 

Create a Conda environment:
```bash
conda create -n ece-6254-project python=3.7.13
conda activate ece-6254-project

```

Install main dependencies:
```bash 
conda install pytorch==1.10.0 torchvision torchtext cpuonly -c pytorch
conda install pip
pip install datasets transformers huggingface_hub
```

**If you are using a GPU install PyTorch accordingly: [Install Pytorch](https://pytorch.org/)**

**See environment.yml for a full list of packages required by this project and install with conda or pip accordingly.**

The following is necessary for using the HuggingFace libraries:
```bash
sudo apt-get install git-lfs
```

### Training the Tesla Stock Tweet Sentiment Analysis Model
This section provides instructions for training the Sentiment Analysis model used to extract sentiment from tweets regarding Tesla stock. We begin with a HuggingFace pre-trained tokenizer and language model. We then train this model to classify the sentiment of tweets by fine tuning on a custom Tesla stock tweet dataset.


#### Run training
You can specify different training options in the `sentiment_training/sentiment_train_options.py` file. When you run the above, the model will be saved according to the `--output_dir` and `--model_name` arguments. 

##### In terminal
Training can be done via the terminal by running the following in the root of this repository:
```bash
python sentiment_train.py 
```

##### In a Jupyter Notebook 

You can run the training on a Jupyter Notebook in the root of this directory. For an example see `sentiment_training_ex.ipynb`




