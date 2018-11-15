import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import time

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
  dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz", 
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      extract=True)
  
  train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
  test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))
  
  return train_df, test_df

def main():
    tf.logging.set_verbosity(tf.logging.ERROR)

    t = time.time()
    train_df, test_df = download_and_load_datasets()
    train_df.head()
    print("Data Loaded, time: %.5fs" % (time.time() - t))


    # Training input on the whole training set with no limit on training epochs.
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["polarity"], batch_size=128, num_epochs=5, shuffle=True)

    # Prediction on the whole training set.
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["polarity"],  batch_size=128, shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df["polarity"],  batch_size=128, shuffle=False)

    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence", 
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")
    print("Sentences embedded, time: %.5fs" % (time.time() - t))

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    """### Training

    Train the estimator for a reasonable amount of steps.
    """

    print("Train")
    # Reduce logging output.
    #tf.logging.set_verbosity(tf.logging.INFO)
    # Training for 1,000 steps means 128,000 training examples with the default
    # batch size. This is roughly equivalent to 5 epochs since the training dataset
    # contains 25,000 examples.
    estimator.train(input_fn=train_input_fn, steps=1000)
    # Reduce logging output.
    #tf.logging.set_verbosity(tf.logging.ERROR)
    print("Training complete, time: %.5fs" % (time.time() - t))
    """# Prediction

    Run predictions for both training and test set.
    """

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    print("Training set accuracy: {accuracy}".format(**train_eval_result))
    print("Test set accuracy: {accuracy}".format(**test_eval_result))


if __name__ == '__main__':
    main()