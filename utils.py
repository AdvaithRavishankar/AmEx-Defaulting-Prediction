import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
import math
from tensorflow.keras.layers import *

def date_converter(x):
    """Qunatifies date for prcoessing 
    
    Args:
        x: the Date formatted to MM/DD/YYYY

    Returns:
        x quantifies using datetime.fromisoformat()
    """
    return datetime.fromisoformat(x).timestamp()

def norm(df, column):
    """Normalizes data with column name

    Args:
        df: DataFrame with the data to be modified
            formatted with index and atleast 1 column
        
        column: column name as a String

    Returns:
        the normalized values of every entry in the dataFrame's column
    """
    return df[column]  / df[column].abs().max()

def get_data_descent(data_path):
    """Gets the Data for processing

    Args:
        data_path: path to csv file (Read as a pandas DataFrame)

    Returns:
        training data and labels as a 1D numpy array
    """
    table = pd.read_csv(data_path)
    table = table.drop("Unnamed: 0", axis=1)
    table = table.fillna(0)
    table["S_2"] = table["S_2"].apply(date_converter)
    table["S_2"] = norm(table,"S_2")
    
    labels = table["target"]
    training = table.drop(["customer_ID","D_63", "D_64", "target"], axis=1)
    
    return training.to_numpy(), np.transpose(np.expand_dims(labels.to_numpy(), axis=0))

def get_data_unflatten(data_path, num_columns=144):
    """Gets the Data for processing

    Args:
        data_path: path to csv file (Read as a pandas DataFrame)
        num_columns: columns to process (needs to be a square number)

    Returns:
        training data fromatted to (no.samples, sqrt(num_columns), sqrt(num_columns)) and
        labels as a 1D array
    """
    table = pd.read_csv(data_path)
    table = table.drop("Unnamed: 0", axis=1)
    table = table.fillna(0)
    table["S_2"] = table["S_2"].apply(date_converter)
    table["S_2"] = norm(table,"S_2")
    
    labels = table["target"]
    training = table.drop(["customer_ID","D_63", "D_64", "target"], axis=1)
    
    training = training.iloc[:, :num_columns].to_numpy()
    training = unflatten(training, int(math.sqrt(num_columns)))
    training = np.expand_dims(training, axis=-1)
    
    return training, np.transpose(np.expand_dims(labels.to_numpy(), axis=0))

def split_data(data, labels, split=1000000):
    """Splits data into train and val splits

    Args:
        data: numpy array for data (no.sample, ...)
        labels: numpy array fro labels (no.sample, ...)
        split: int for the distribution of train and val

    Returns:
        Tenor objects with return order 
        train_data, train_labels, val_data, val_labels
    """
    train_data, train_labels, val_data, val_labels = data[:split], labels[:split], data[split:], labels[split:]

    train_data = tf.convert_to_tensor(train_data, dtype=tf.float64)
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float64)

    val_data = tf.convert_to_tensor(val_data, dtype=tf.float64)
    val_labels = tf.convert_to_tensor(val_labels, dtype=tf.float64)

    return train_data, train_labels, val_data, val_labels
