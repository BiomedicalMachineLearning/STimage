import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Lambda
from tensorflow.keras.models import Model


class PrinterCallback(tf.keras.callbacks.Callback):

    # def on_train_batch_begin(self, batch, logs=None):
    #     # Do something on begin of training batch

    def on_epoch_end(self, epoch, logs=None):
        print('EPOCH: {}, Train Loss: {}, Val Loss: {}'.format(epoch,
                                                               logs['loss'],
                                                               logs['val_loss']))

    def on_epoch_begin(self, epoch, logs=None):
        print('-' * 50)
        print('STARTING EPOCH: {}'.format(epoch))


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


def CNN_NB_trainable(tile_shape):
    tile_input = Input(shape=tile_shape, name="tile_input")
    resnet_base = ResNet50(input_tensor=tile_input, weights='imagenet', include_top=False)
    stage_5_start = resnet_base.get_layer("conv5_block1_1_conv")
    for i in range(resnet_base.layers.index(stage_5_start)):
        resnet_base.layers[i].trainable = False

    cnn = resnet_base.output
    cnn = GlobalAveragePooling2D()(cnn)
    cnn = Dropout(0.5)(cnn)
    cnn = Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l1(0.1),
                activity_regularizer=keras.regularizers.l2(0.1))(cnn)
    # cnn = Dense(256, activation='relu')(cnn)
    outputs = Dense(2)(cnn)
    distribution_outputs = Lambda(negative_binomial_layer)(outputs)
    model = Model(inputs=tile_input, outputs=distribution_outputs)

    # optimizer = tf.keras.optimizers.RMSprop(0.0001)
    optimizer = tf.keras.optimizers.Adam()

    model.compile(loss=negative_binomial_loss,
                  optimizer=optimizer,
                  metrics=[negative_binomial_loss])
    return model


def CNN_NB_multiple_genes(tile_shape, n_genes, cnnbase="resnet50", ft=False):
    tile_input = Input(shape=tile_shape, name="tile_input")
    if cnnbase == "resnet50":
        cnn_base = ResNet50(input_tensor=tile_input, weights='imagenet', include_top=False)
    elif cnnbase == "vgg16":
        cnn_base = VGG16(input_tensor=tile_input, weights='imagenet', include_top=False)
    elif cnnbase == "inceptionv3":
        cnn_base = InceptionV3(input_tensor=tile_input, weights='imagenet', include_top=False)
    elif cnnbase == "mobilenetv2":
        cnn_base = MobileNetV2(input_tensor=tile_input, weights='imagenet', include_top=False)
    elif cnnbase == "densenet121":
        cnn_base = DenseNet121(input_tensor=tile_input, weights='imagenet', include_top=False)
    elif cnnbase == "xception":
        cnn_base = Xception(input_tensor=tile_input, weights='imagenet', include_top=False)
    #     stage_5_start = resnet_base.get_layer("conv5_block1_1_conv")
    #     for i in range(resnet_base.layers.index(stage_5_start)):
    #         resnet_base.layers[i].trainable = False

    if not ft:
        for i in cnn_base.layers:
            i.trainable = False
    cnn = cnn_base.output
    cnn = GlobalAveragePooling2D()(cnn)
    #     cnn = Dropout(0.5)(cnn)
    #     cnn = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
    #                 activity_regularizer=tf.keras.regularizers.l2(0.01))(cnn)
    # cnn = Dense(256, activation='relu')(cnn)
    output_layers = []
    for i in range(n_genes):
        output = Dense(2)(cnn)
        output_layers.append(Lambda(negative_binomial_layer, name="gene_{}".format(i))(output))

    model = Model(inputs=tile_input, outputs=output_layers)
    #     losses={}
    #     for i in range(8):
    #         losses["gene_{}".format(i)] = negative_binomial_loss(i)
    #     optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.keras.optimizers.Adam(1e-5)
    model.compile(loss=negative_binomial_loss,
                  optimizer=optimizer)
    return model
