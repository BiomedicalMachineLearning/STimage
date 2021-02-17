import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy


def negative_binomial_layer(x):
    """
    Lambda function for generating negative binomial parameters
    n and p from a Dense(2) output.
    Assumes tensorflow 2 backend.

    Usage
    -----
    outputs = Dense(2)(final_layer)
    distribution_outputs = Lambda(negative_binomial_layer)(outputs)

    Parameters
    ----------
    x : tf.Tensor
        output tensor of Dense layer

    Returns
    -------
    out_tensor : tf.Tensor

    """

    # Get the number of dimensions of the input
    num_dims = len(x.get_shape())

    # Separate the parameters
    n, p = tf.unstack(x, num=2, axis=-1)

    # Add one dimension to make the right shape
    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)

    # Apply a softplus to make positive
    n = tf.keras.activations.softplus(n)

    # Apply a sigmoid activation to bound between 0 and 1
    p = tf.keras.activations.sigmoid(p)

    # Join back together again
    out_tensor = tf.concat((n, p), axis=num_dims - 1)

    return out_tensor


def negative_binomial_loss(y_true, y_pred):
    """
    Negative binomial loss function.
    Assumes tensorflow backend.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        n and p values of predicted distribution.

    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """

    # Separate the parameters
    n, p = tf.unstack(y_pred, num=2, axis=-1)

    # Add one dimension to make the right shape
    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)

    # Calculate the negative log likelihood
    nll = (
            tf.math.lgamma(n)
            + tf.math.lgamma(y_true + 1)
            - tf.math.lgamma(n + y_true)
            - n * tf.math.log(p)
            - y_true * tf.math.log(1 - p)
    )

    return nll


def CNN_NB_model():
    inputs = Input(shape=(2048,))
    outputs = Dropout(0.5)(inputs)
    #     outputs = Dense(512,)(outputs)
    #     outputs = Dense(256, activation='relu')(inputs)
    #     outputs = Dropout(0.5)(outputs)
    outputs = Dense(2)(outputs)
    distribution_outputs = Lambda(negative_binomial_layer)(outputs)

    model = Model(inputs=inputs, outputs=distribution_outputs)

    optimizer = tf.keras.optimizers.RMSprop(0.0001)
    #     optimizer = tf.keras.optimizers.Adam()

    model.compile(loss=negative_binomial_loss,
                  optimizer=optimizer,
                  metrics=[negative_binomial_loss])
    return model


def CNN_linear_model():
    inputs = Input(shape=(2048,))
    outputs = Dropout(0.6)(inputs)
    #     outputs = Dense(512,)(outputs)
    #     outputs = Dense(256, activation='relu')(inputs)
    #     outputs = Dropout(0.5)(outputs)
    outputs = Dense(1, activation='linear')(inputs)

    model = Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.RMSprop(0.0001)
    #     optimizer = tf.keras.optimizers.Adam()

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mse'])
    return model