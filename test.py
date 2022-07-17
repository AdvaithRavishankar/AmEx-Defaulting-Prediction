import tensorflow as tf
import pandas as pd
import numpy as np
import math
from datetime import datetime
import utils.py

def test_descent(data_path, model_path, output_path):
    """Generates Predictions for test data

    Args:
        data_path: path to test data
        model_path: path to model to evluate data
        output_path: path for output predictions (must end with .csv)
    """
    data = get_data_descent(data_path=data_path)
    data = tf.convert_to_tensor(data)

    model = tf.keras.models.load_model(model_path)

    evals = model.predict(data)
    df = pd.DataFrame()
    df["customer_ID"] = names
    df["prediction"] = evals
    df = df.set_index("customer_ID")
    df.to_csv(output_path)

def test_unflatten(data_path, model_path, output_path, num_columns=144):
    """Generates Predictions for test data

    Args:
        data_path: path to test data
        model_path: path to model to evluate data
        output_path: path for output predictions (must end with .csv)
        num_columns: num of columns to process (must be a square number)

    """
    data = get_data_unflatten(data_path=data_path, num_columns=num_columns)
    data = tf.convert_to_tensor(data)

    model = tf.keras.models.load_model(model_path)

    evals = model.predict(data)
    df = pd.DataFrame()
    df["customer_ID"] = names
    df["prediction"] = evals
    df = df.set_index("customer_ID")
    df.to_csv(output_path)

