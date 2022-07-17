import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
    """Creates a convoltuional block.
    Args:
        inputs: the data for the layer
        num_filters: the filter size for the convolution
    
    Returns:
        a convolution block
    """
    x = Conv2D(num_filters, 3, padding ="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding ="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    """Creates an encoding block
    Args:
        inputs: the data for the layer
        num_filters: the filter size for the convolution
    
    Returns:
        An encoding block
    """
    x = conv_block(inputs, num_filters)
    p = MaxPool2D((2,2))(x)

    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """Creates an decoding block
    Args:
        inputs: the data for the layer
        num_filters: the filter size for the convolution
    
    Returns:
        An encoding block
    """
    x = Conv2DTranspose(num_filters, (2,2), strides=(2,2), padding="same")(inputs)
    x = Concatenate(axis=3)([x, skip_features])
    x = conv_block(x, num_filters)

    return x

def build_unet(input_shape):
    """Initializes a Simple Unit
    Args:
        input_shape: the size of the image
        num_channels: the number of output channels
    Returns:
        a built U-Net model 
    """

    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 16)

    """ Bridge """

    b1 = conv_block(p1, 64)

    """ Decoder """
    d1 = decoder_block(b1, s1, 16)

    """ Output """

    unet = Conv2D(32, (1,1), padding="same")(d1)
    
    flat = Flatten()(unet)
    
    dense1 = Dense(80, activation="relu")(flat)
    dense2 = Dense(40, activation="relu")(dense1)
    dense3 = Dense(10, activation="relu")(dense2)
    
    outputs = Dense(1)(dense3)


    model = Model(inputs, outputs, name="U-Net")

    return model

def build_descent(input_shape):
    """Makes a decending neural network

    args:
        input_shape: number of column variables network should train on

    Returns:
        Model architure with weights
    """
    NN_model = tf.keras.Sequential()

    # The Input Layer :
    NN_model.add(Dense(input_shape, input_dim = input_shape, activation="relu"))

    #Hidden Layers
    NN_model.add(Dense(80,activation='relu'))

    NN_model.add(Dense(40,activation='relu'))

    NN_model.add(Dense(20,activation='relu'))

    # The Output Layer :
    NN_model.add(Dense(1))

    return NN_model

def unflatten(data, dim):
    """Transforms a 1D array to a 2d image array
       formatted to (dim, dim)

    Args:   
        data: 1D Array to be reshaped
        dim: dim of out array

    Returns:
        unflattened arrat formatted to (dim, dim)
    """
    reshaped = []
    for i in data:
        reshaped.append(i.reshape(dim, -1).T)

    return np.array(reshaped)