import pandas as pd
import numpy as np
import random
from datetime import datetime
import model_utils
import utils

def descent_pipeline():
    """Starts Training For Descent Network"""
    data_path = #Depends on relative path

    data, labels = get_data_descent(data_path=data_path)

    model = build_descent()

    train_data, train_labels, val_data, val_labels = split_data(data, labels)

    initial_lr = 1e-3

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr, amsgrad=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

    callbacks = [
             tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_delta=0.001),
             tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=0.001) 
    ]

    # Approx Training Time 1 hour
    history = model.fit(
        train_data,
        train_labels,
        validation_data=[val_data, val_labels],
        epochs=200,
        callbacks=callbacks
    )

def image_pipeline():
    """Starts Training For Imgae-Styled Network"""
    data_path = #Depends on relative path

    data, labels = get_data_unflatten(data_path=data_path)

    model = build_unet()

    train_data, train_labels, val_data, val_labels = split_data(data, labels)

    initial_lr = 1e-3

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr, amsgrad=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])

    callbacks = [
             tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_delta=0.001),
             tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=0.001) 
    ]

    # Approx Training Time 6 hour
    history = model.fit(
        train_data,
        train_labels,
        validation_data=[val_data, val_labels],
        epochs=200,
        callbacks=callbacks
    )
