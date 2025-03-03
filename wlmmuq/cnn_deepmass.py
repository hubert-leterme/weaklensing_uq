"""
Modified version of the 'cnn_keras.py' module from DeepMass
https://github.com/NiallJeffrey/DeepMass

"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.layers import concatenate, AveragePooling2D


class BaseL2RegLoss(keras.losses.Loss):

    def __init__(self, l2_lambda=1e-4, offset=0., **kwargs):
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

    def __init__(self, trainable=False, offset=0., **kwargs):
        super().__init__(trainable=trainable, **kwargs)
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


class BaseModel:

    def __init__(
            self, map_size, learning_rate, loss='mse', l2_lambda=1e-4, offset=0.
    ):
        """
        Initialization
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        :param loss: loss function: 'mse', 'mae', 'l2reg_mse' or 'l2reg_mae'
        :param l2_lambda: regularization parameter
        :param offset: mean value of the convergence maps. Default = 0.
        """
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.loss = loss
        self.l2_lambda = l2_lambda
        self.offset = offset

        self.inputs, self.outputs = self._init_model()


    def _init_model(self):
        raise NotImplementedError


    def model(self):

        out = keras.models.Model(self.inputs, self.outputs)
        out.summary()

        if self.loss in ('mse', 'mae'):
            loss_fun = self.loss
        elif self.loss == 'l2reg_mse':
            loss_fun = L2RegMSE(l2_lambda=self.l2_lambda, offset=self.offset)
        elif self.loss == 'l2reg_mae':
            loss_fun = L2RegMAE(l2_lambda=self.l2_lambda, offset=self.offset)
        else:
            raise ValueError

        if self.learning_rate is None:
            out.compile(optimizer='adam', loss=loss_fun)
        else:
            out.compile(
                optimizer=keras.optimizers.Adam(lr=self.learning_rate),
                loss=loss_fun
            )

        return out


class SimpleModel(BaseModel):
    """
    A CNN class that creates a simple denoiser
    """

    def _init_model(self):

        input_img = Input(shape=(self.map_size, self.map_size, 1))

        filters = 32

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_img)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        final = Conv2D(1, (3, 3), activation='sigmoid', padding='same', kernel_initializer='he_normal')(x)

        return input_img, final


class UNet(BaseModel):
    """
    A CNN class that creates a denoising Unet
    """

    def __init__(
            self, *args, in_channels=1, out_channels=1, mean_centering=False,
            use_bias=True, **kwargs
    ):
        """
        Initialisation
        :param map_size: size of square image (there are map_size**2 pixels)
        :param learning_rate: learning rate for the optimizer
        :param in_channels: number of input channels. Default = 1
        :param out_channels: number of output channels. Default = 1
        :param mean_centering: whether to apply mean centering at the output.
            Default = False
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mean_centering = mean_centering
        self.use_bias = use_bias
        super().__init__(*args, **kwargs)


    def _init_model(self):

        input_img = Input(shape=(self.map_size, self.map_size, self.in_channels))

        x1 = Conv2D(
            16, 3, activation='relu', padding='same', kernel_initializer='he_normal',
            use_bias=self.use_bias
        )(input_img)
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
        output = Conv2D(self.out_channels, 1, activation='sigmoid')(x7)

        if self.mean_centering:
            output = MeanCentering(self.offset)(output)

        return input_img, output
