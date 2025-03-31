"""
Modified version of the 'cnn_keras.py' module from DeepMass
https://github.com/NiallJeffrey/DeepMass

"""
import warnings

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import concatenate, AveragePooling2D, Add, Multiply

from . import LOSS, L2_LAMBDA

class BaseL2RegLoss(keras.losses.Loss):

    def __init__(self, l2_lambda=L2_LAMBDA, offset=0., **kwargs):
        super().__init__(**kwargs)
        self.l2_lambda = l2_lambda
        self.offset = offset

    def call(self, y_true, y_pred):
        df = self._data_fidelity(y_true, y_pred)
        l2_output = tf.reduce_mean(tf.square(y_pred - self.offset))
        return df + self.l2_lambda * l2_output

    def _data_fidelity(self, y_true, y_pred):
        raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            "l2_lambda": self.l2_lambda,
            "offset": self.offset
        })
        return config


class L2RegMSE(BaseL2RegLoss):
    def _data_fidelity(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))


class L2RegMAE(BaseL2RegLoss):
    def _data_fidelity(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))


class MeanCentering(keras.layers.Layer):

    def __init__(self, offset=0., **kwargs):
        super().__init__(**kwargs)
        self.offset = offset

    def call(self, tensor):
        tensor -= self.offset
        tensor -= tf.reduce_mean(tensor, axis=[1, 2], keepdims=True)
        tensor += self.offset
        return tensor

    def get_config(self):
        config = super().get_config()
        config.update({
            "offset": self.offset
        })
        return config


class Square(keras.layers.Layer):
    def call(self, tensor):
        return tf.square(tensor)


class BaseModel(keras.models.Model):

    def __init__(self, map_size, offset=0.):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param offset: mean value of the convergence maps (for mean centering).
            Default = 0.
        """
        self.map_size = map_size
        self.offset = offset
        kwargs = self._init_model()
        super().__init__(**kwargs)

    def _init_model(self) -> dict:
        raise NotImplementedError


class SimpleModel(BaseModel):
    """
    A CNN class that creates a simple denoiser
    """

    def _init_model(self) -> dict:

        inp = Input(shape=(self.map_size, self.map_size, 1))

        filters = 32

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inp)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        final = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)

        out_dict = {'inputs': inp, 'outputs': final}

        return out_dict


class UNet(BaseModel):
    """
    A CNN class that creates a denoising Unet
    """

    def __init__(
            self, *args, in_channels=1, out_channels=1, mean_centering=False,
            use_bias=True, sigmoid_activation=False, **kwargs
    ):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param offset: mean value of the convergence maps (for mean centering).
            Default = 0.
        :param in_channels: number of input channels. Default = 1
        :param out_channels: number of output channels. Default = 1
        :param mean_centering: whether to apply mean centering at the output.
            Default = False
        :param use_bias: whether to use bias in the convolutional and batch
            normalization layers. Default = True
        :param sigmoid_activation: whether to apply a sigmoid activation function
            at the output. Default = True
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mean_centering = mean_centering
        self.use_bias = use_bias
        if sigmoid_activation:
            self.activation = 'sigmoid'
        else:
            self.activation = None
        super().__init__(*args, **kwargs)


    def _init_model(self) -> dict:

        inp = Input(shape=(self.map_size, self.map_size, self.in_channels))

        x1 = Conv2D(
            16, 3, activation='relu', padding='same', kernel_initializer='he_normal',
            use_bias=self.use_bias
        )(inp)
        x1 = BatchNormalization(center=self.use_bias)(x1)

        pool1 = AveragePooling2D(pool_size=(2, 2))(x1)
        x2 = Conv2D(
            32, 3, activation='relu', padding='same', kernel_initializer='he_normal',
            use_bias=self.use_bias
        )(pool1)
        x2 = BatchNormalization(center=self.use_bias)(x2)

        pool2 = AveragePooling2D(pool_size=(2, 2))(x2)
        x3 = Conv2D(
            64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
            use_bias=self.use_bias
        )(pool2)
        x3 = BatchNormalization(center=self.use_bias)(x3)

        pool3 = AveragePooling2D(pool_size=(2, 2))(x3)
        x4 = Conv2D(
            64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
            use_bias=self.use_bias
        )(pool3)
        x4 = BatchNormalization(center=self.use_bias)(x4)

        pool_deep = AveragePooling2D(pool_size=(2, 2))(x4)
        xdeep = Conv2D(
            64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
            use_bias=self.use_bias
        )(pool_deep)
        xdeep = BatchNormalization(center=self.use_bias)(xdeep)

        updeep = UpSampling2D((2, 2))(xdeep)
        mergedeep = concatenate([x4, updeep], axis=3)

        xdeep2 = Conv2D(
            64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
            use_bias=self.use_bias
        )(mergedeep)
        xdeep2 = BatchNormalization(center=self.use_bias)(xdeep2)

        up5 = UpSampling2D((2, 2))(xdeep2)
        merge5 = concatenate([x3, up5], axis=3)
        merge5 = BatchNormalization(center=self.use_bias)(merge5)

        x5 = Conv2D(
            64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
            use_bias=self.use_bias
        )(merge5)

        up6 = UpSampling2D((2, 2))(x5)
        merge6 = concatenate([x2, up6], axis=3)
        merge6 = BatchNormalization(center=self.use_bias)(merge6)

        x6 = Conv2D(
            32, 3, activation='relu', padding='same', kernel_initializer='he_normal',
            use_bias=self.use_bias
        )(merge6)

        up7 = UpSampling2D((2, 2))(x6)
        merge7 = concatenate([x1, up7], axis=3)
        merge7 = BatchNormalization(center=self.use_bias)(merge7)

        x7 = Conv2D(
            16, 3, activation='relu', padding='same', kernel_initializer='he_normal',
            use_bias=self.use_bias
        )(merge7)
        out = Conv2D(self.out_channels, 1, activation=self.activation)(x7)

        inp, out = self._postprocess(inp, out)

        if self.mean_centering:
            out = MeanCentering(self.offset)(out)

        out_dict = {'inputs': inp, 'outputs': out}

        return out_dict


    def _postprocess(self, inp, out):
        return inp, out # Identity in the base class


class UNetFromScore(UNet):
    """
    A U-Net model that incorporates a scalar multiplier named sigma,
    such that the output is computed as: out = inp + sigma**2 * UNet(inp).
    According to Tweedie's formula, assuming that:
    - input images are corrupted by a Gaussian noise with standard deviation given by sigma;
    - UNet computes the gradient of the log-prior PDF of the noisy images,
    then UNetFromScore corresponds to an MMSE denoiser.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sigmoid_activation:
            warnings.warn("Sigmoid activation will lead to unexpected behavior")

    def _postprocess(self, inp, out):
        # Argument out estimates the score of the log-prior PDF of the noisy images
        sigma = Input(shape=(1, 1, 1))
        var = Square()(sigma)
        minusnoise = Multiply()([var, out])
        out = Add()([inp, minusnoise])

        return (inp, sigma), out


def compile_kerasmodel(model, loss=LOSS, learning_rate=None, **kwargs):
    """
    :param model: keras model
    :param loss: loss function: 'mse', 'mae', 'l2reg_mse' or 'l2reg_mae'
    :param learning_rate: learning rate for the optimizer
    :param l2_lambda: regularization parameter
    :param offset: mean value of the convergence maps (for mean centering).
        Default = 0.
    """
    if loss in ('mse', 'mae'):
        loss_fun = loss
    elif loss == 'l2reg_mse':
        loss_fun = L2RegMSE(**kwargs)
    elif loss == 'l2reg_mae':
        loss_fun = L2RegMAE(**kwargs)
    else:
        raise ValueError

    if learning_rate is None:
        model.compile(optimizer='adam', loss=loss_fun)
    else:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fun
        )
