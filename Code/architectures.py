'''
Creates WNET architecture as a keras model and trains it.
'''
# %% IMPORTS
import math
import tensorflow as tf
from tensorflow import keras
from utils import losses
from utils.misc import upper_round
# %% CLASSES AND FUNCTIONS


class UNET():
    '''
    Creates keras model with UNET architecture.
    '''

    def __init__(self, n_filters, input_shape, optimizer_name, learning_rate, loss_name):
        '''
        Set network parameters.

        Arguments:
        n_filters {int} -- number of initial convolutional filters
        input_shape {tuple} -- input shape of the image (width, height)
        optimizer_name {str} -- name of the optimizer ('adam', 'adadelta', etc)
        learning_rate {float} -- learning rate fot the optimizer
        loss_name {str} -- name of the loss to minimize when training
        '''
        self.base = 2**5  # input image will be padded with zeros so the width/height is multiple of this number
        self.n_filters = n_filters
        self.input_height, self.input_width, self.input_channels = input_shape
        self.learning_rate = learning_rate
        if optimizer_name == 'adam':
            self.optimizer = tf.optimizers.Adam(self.learning_rate)

        if loss_name == 'iou':
            self.loss_fn = losses.iou_coef
        self.model = None

        self.width_padding = upper_round(
            self.base, self.input_width)
        self.height_padding = upper_round(
            self.base, self.input_height)

        self.padding = (math.ceil(self.height_padding/2),
                        math.ceil(self.width_padding/2))

    def contract_conv_block(self, inputs, filters, bottom=False):
        '''
        Contracting convolutional block. Performs two succesive convolutions with kernel shape 3x3.
        Dropout of 0.1, batch normalization and 'relu' activation are applied.
        Max pooling of 2x2 is applied.
        Returns the layer before and after pooling to make use of skip layers.

        Arguments:
        inputs {keras layer} -- input layer
        filters {int} -- number of convolutional filters

        Keyword Arguments:
        bottom {bool}: checks if its the last contracting layer (default: {False})

        Returns:
        [conv, pool] {keras layer, keras layer} -- [last convolutional layer / skip layer , output layer]
        '''
        if filters == self.n_filters or self.input_channels == 1:
            conv_layer = keras.layers.Conv2D
        else:
            # perform separable convolution if image has multiple channels (RGB)
            conv_layer = keras.layers.SeparableConv2D

        conv = conv_layer(filters, (3, 3), activation='relu',
                          kernel_initializer='he_normal', padding='same')(inputs)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.Dropout(0.1)(conv)
        conv = conv_layer(filters, (3, 3), activation='relu',
                          kernel_initializer='he_normal', padding='same')(conv)
        conv = keras.layers.BatchNormalization()(conv)
        if bottom:
            return conv
        pool = keras.layers.MaxPooling2D((2, 2))(conv)
        return conv, pool

    def expand_conv_block(self, inputs, filters, concat_layer):
        '''
        Expanding convolutional block. Performs transposed convolution of kernel shape 2x2 and two succesive convolutions with kernel shape 3x3.
        Uses skip layers returned from contract_conv_block and concatenates after transposed convolution.
        Dropout of 0.2, batch normalization and 'relu' activation are applied.
        Max pooling of 2x2 is applied.

        Arguments:
        inputs {keras layer} -- input layer
        filters {int} -- number of convolutional filters
        concat_layer {keras layer} -- last convolutional layer from corresponding contracting path

        Returns:
        [conv] {keras layer} -- output layer
        '''
        if filters == self.n_filters or self.input_channels == 1:
            conv_layer = keras.layers.Conv2D
        else:
            conv_layer = keras.layers.SeparableConv2D
        upsample = keras.layers.Conv2DTranspose(filters//2, (2, 2), strides=(2, 2),
                                                padding='same')(inputs)
        upsample = keras.layers.concatenate([upsample, concat_layer])
        conv = conv_layer(filters, (3, 3), activation='relu',
                          kernel_initializer='he_normal', padding='same')(upsample)
        conv = keras.layers.BatchNormalization()(conv)
        conv = keras.layers.Dropout(0.2)(conv)
        conv = conv_layer(filters, (3, 3), activation='relu',
                          kernel_initializer='he_normal', padding='same')(conv)
        conv = keras.layers.BatchNormalization()(conv)
        return conv

    def unet(self, inputs):
        '''
        Builds U-Net architecture upon 4 contracting blocks each doubling number the of initial filters and 4 expanding blocks halving the number
        of filters. Skip connections are returned by contracting blocks to then concatenate with expanding blocks.

        Arguments:
        inputs [keras layer] -- input layer

        Returns:
        output [keras layer] -- output layer
        '''
        # UNET ENCODER
        # CONTRACTING PATH
        # BLOCK 1
        conv1, pool1 = self.contract_conv_block(inputs, self.n_filters)
        # BLOCK 2
        conv2, pool2 = self.contract_conv_block(pool1, self.n_filters*2)
        # BLOCK 3
        conv3, pool3 = self.contract_conv_block(pool2, self.n_filters*4)
        # BLOCK 4
        conv4, pool4 = self.contract_conv_block(pool3, self.n_filters*8)
        # BLOCK 5 (BOTTOM)
        conv5 = self.contract_conv_block(pool4, self.n_filters*16, bottom=True)
        # EXPANDING PATH
        # BLOCK 6
        conv6 = self.expand_conv_block(conv5, self.n_filters*8, conv4)
        # BLOCK 7
        conv7 = self.expand_conv_block(conv6, self.n_filters*4, conv3)
        # BLOCK 8
        conv8 = self.expand_conv_block(conv7, self.n_filters*2, conv2)
        # BLOCK 9
        output = self.expand_conv_block(conv8, self.n_filters, conv1)
        return output

    def create_model(self):
        '''
        Adds the output layer to the model with 'sigmoid' activation, compiles it and returns a keras model object.

        Returns:
        _model {keras model} -- keras model with U-Net architecture
        '''
        inputs = keras.layers.Input(
            (self.input_height, self.input_width, self.input_channels))
        pad_input = keras.layers.ZeroPadding2D(self.padding)(inputs)

        conv9 = self.unet(pad_input)
        cropped = keras.layers.Cropping2D(self.padding)(conv9)
        output = keras.layers.Conv2D(
            1, (1, 1), activation='sigmoid')(cropped)

        _model = tf.keras.Model(inputs=[inputs], outputs=[output])
        _model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        self.model = _model

        return self.model


class DECONVNET(UNET):
    '''
    Creates keras model with DeconvNet architecture.
    '''

    def __init__(self, n_filters, input_shape, optimizer_name, learning_rate, loss_name):
        '''
        Set network parameters.

        Arguments:
        n_filters {int} -- number of initial convolutional filters
        input_shape {tuple} -- input shape of the image (width, height)
        optimizer_name {str} -- name of the optimizer ('adam', 'adadelta', etc)
        learning_rate {float} -- learning rate for the optimizer
        loss_name {str} -- name of the loss to minimize when training
        '''
        UNET.__init__(self, n_filters, input_shape,
                      optimizer_name, learning_rate, loss_name)

    def conv2dblock(self, inputs, filters, depth):
        '''
        Contracting convolutional block. Performs a number of succesive convolutions of kernel shape 3x3. Batch normalization and 'relu'
        activation are applied.

        Arguments:
        inputs {keras layer} -- input layer
        filters {int} -- number of convolutional filters
        depth {int} -- number of succesive convolutions

        Returns:
        [conv] {keras layer} -- [output layer]
        '''
        for i in range(1, depth + 1):
            if i == 1:
                conv = keras.layers.Conv2D(
                    filters, (3, 3), padding='same', use_bias=False)(inputs)
            else:
                conv = keras.layers.Conv2D(
                    filters, (3, 3), padding='same', use_bias=False)(conv)

            conv = keras.layers.BatchNormalization()(conv)
            conv = keras.layers.Activation('relu')(conv)
        conv = keras.layers.MaxPooling2D((2, 2))(conv)
        return conv

    def conv2dtransposeblock(self, inputs, filters, depth):
        '''
        Expanding convolutional block. Performs transposed convolution of kernel shape 2x2 and a number of succesive convolutions with kernel
        shape 3x3.
        Uses skip layers returned from contract_conv_block and concatenates after transposed convolution.
        Dropout of 0.2, batch normalization and 'relu' activation are applied. Max pooling of 2x2 is applied.

        Arguments:
        inputs {keras layer} -- input layer
        filters {int} -- number of convolutional filters
        depth {int} -- number of succesive convolutions

        Returns:
        [conv] {keras layer} -- output layer
        '''
        deconv = keras.layers.Conv2DTranspose(
            filters, (2, 2), strides=(2, 2), use_bias=False)(inputs)
        deconv = keras.layers.BatchNormalization()(deconv)
        deconv = keras.layers.Activation('relu')(deconv)
        for i in range(1, depth + 1):
            if i == 1:
                conv2d = keras.layers.Conv2D(
                    filters, (3, 3), padding='same', use_bias=False)(deconv)
            else:
                conv2d = keras.layers.Conv2D(
                    filters, (3, 3), padding='same', use_bias=False)(deconv)

            deconv = keras.layers.BatchNormalization()(conv2d)
            deconv = keras.layers.Activation('relu')(conv2d)
        return deconv

    def deconvnet(self, inputs):
        '''
        Builds DeconvNet architecture upon 4 contracting blocks each doubling the number of initial filters and 4 expanding blocks halving
        the number of filters.

        Arguments:
        inputs [keras layer] -- input layer

        Returns:
        output [keras layer] -- output layer
        '''
        b_1 = self.conv2dblock(inputs, self.n_filters, 2)
        b_2 = self.conv2dblock(b_1, self.n_filters*2, 2)
        b_3 = self.conv2dblock(b_2, self.n_filters*4, 3)
        b_4 = self.conv2dblock(b_3, self.n_filters*8, 3)
        b_5 = self.conv2dblock(b_4, self.n_filters*8, 3)

        fc6 = keras.layers.Conv2D(self.n_filters*8, ((self.input_height+self.height_padding)//self.base, (self.input_width+self.width_padding)//self.base),
                                  use_bias=False, padding='valid')(b_5)  # 4096
        fc6 = keras.layers.BatchNormalization()(fc6)
        fc6 = keras.layers.Activation('relu')(fc6)

        fc7 = keras.layers.Conv2D(
            self.n_filters*8, 1, use_bias=False, padding='valid')(fc6)  # 4096
        fc7 = keras.layers.BatchNormalization()(fc7)
        fc7 = keras.layers.Activation('relu')(fc7)

        deconv = keras.layers.Conv2DTranspose(
            self.n_filters*8, ((self.input_height+self.height_padding)//self.base, (self.input_width+self.width_padding)//self.base), use_bias=False)(fc7)

        deconv = keras.layers.BatchNormalization()(deconv)
        deconv = keras.layers.Activation('relu')(deconv)

        b_8 = self.conv2dtransposeblock(
            deconv, self.n_filters*8, 3)
        b_9 = self.conv2dtransposeblock(
            b_8, self.n_filters*8, 3)
        b_10 = self.conv2dtransposeblock(
            b_9, self.n_filters*4, 3)
        b_11 = self.conv2dtransposeblock(
            b_10, self.n_filters*2, 2)
        output = self.conv2dtransposeblock(
            b_11, self.n_filters, 2)

        return output

    def create_model(self):
        '''
        Adds the output layer to the model with 'sigmoid' activation, compiles it and returns a keras model object.

        Returns:
        _model [keras model] -- keras model with DeconvNet architecture
        '''
        inputs = keras.layers.Input(
            (self.input_height, self.input_width, self.input_channels))
        pad_input = keras.layers.ZeroPadding2D(self.padding)(inputs)

        conv9 = self.deconvnet(pad_input)
        cropped = keras.layers.Cropping2D(self.padding)(conv9)
        output = keras.layers.Conv2D(
            1, (1, 1), activation='sigmoid')(cropped)

        _model = tf.keras.Model(inputs=[inputs], outputs=[output])
        _model.compile(optimizer=self.optimizer, loss=self.loss_fn)

        self.model = _model

        return self.model


class MULTIRES(UNET):
    '''
    Creates keras model with MultiResUnet architecture.
    '''

    def __init__(self, n_filters, input_shape, optimizer_name, learning_rate, loss_name):
        '''
        Set network parameters.

        Arguments:
        n_filters {int} -- number of initial convolutional filters
        input_shape {tuple} -- input shape of the image (width, height)
        optimizer_name {str} -- name of the optimizer ('adam', 'adadelta', etc)
        learning_rate {float} -- learning rate fot the optimizer
        loss_name {str} -- name of the loss to minimize when training
        '''
        UNET.__init__(self, n_filters, input_shape,
                      optimizer_name, learning_rate, loss_name)

    @staticmethod
    def conv2d_bn(input_layer, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu'):
        '''
        2D Convolutional layers

        Arguments:
            input_layer {keras layer} -- input layer
            filters {int} -- number of filters
            num_row {int} -- number of rows in filters
            num_col {int} -- number of columns in filters

        Keyword Arguments:
            padding {str} -- mode of padding (default: {'same'})
            strides {tuple} -- stride of convolution operation (default: {(1, 1)})
            activation {str} -- activation function (default: {'relu'})
            name {str} -- name of the layer (default: {None})

        Returns:
            [keras layer] -- [output layer]
        '''

        input_layer = keras.layers.Conv2D(filters, (num_row, num_col),
                                          strides=strides, padding=padding, use_bias=False)(input_layer)
        input_layer = keras.layers.BatchNormalization(
            axis=3, scale=False)(input_layer)

        if activation is None:
            return input_layer

        output_layer = keras.layers.Activation(activation)(input_layer)

        return output_layer

    @staticmethod
    def trans_conv2d_bn(input_layer, filters, num_row, num_col, padding='same', strides=(2, 2)):
        '''
        2D Transposed Convolutional layers

        Arguments:
            input_layer {keras layer} -- input layer
            filters {int} -- number of filters
            num_row {int} -- number of rows in filters
            num_col {int} -- number of columns in filters

        Keyword Arguments:
            padding {str} -- mode of padding (default: {'same'})
            strides {tuple} -- stride of convolution operation (default: {(2, 2)})
            name {str} -- name of the layer (default: {None})

        Returns:
            [keras layer] -- [output layer]
        '''

        input_layer = keras.layers.Conv2DTranspose(
            filters, (num_row, num_col), strides=strides, padding=padding)(input_layer)
        output_layer = keras.layers.BatchNormalization(
            ainput_layeris=3, scale=False)(input_layer)

        return output_layer

    def multi_res_block(self, n_filters, inp, alpha=1.67):
        '''
        MultiRes Block

        Arguments:
            n_filters {int} -- Number of filters in a corrsponding UNet stage
            inp {keras layer} -- input layer

        Returns:
            [keras layer] -- [output layer]
        '''

        weights = alpha * n_filters

        shortcut = inp

        shortcut = self.conv2d_bn(shortcut, int(weights*0.167) + int(weights*0.333) +
                                  int(weights*0.5), 1, 1, activation=None, padding='same')

        conv3x3 = self.conv2d_bn(inp, int(weights*0.167), 3, 3,
                                 activation='relu', padding='same')

        conv5x5 = self.conv2d_bn(conv3x3, int(weights*0.333), 3, 3,
                                 activation='relu', padding='same')

        conv7x7 = self.conv2d_bn(conv5x5, int(weights*0.5), 3, 3,
                                 activation='relu', padding='same')

        out = keras.layers.concatenate([conv3x3, conv5x5, conv7x7], axis=3)
        out = keras.layers.BatchNormalization(axis=3)(out)

        out = keras.layers.add([shortcut, out])
        out = keras.layers.Activation('relu')(out)
        out = keras.layers.BatchNormalization(axis=3)(out)

        return out

    def res_path(self, filters, length, inp):
        '''
        res_path

        Arguments:
            filters {int} -- [description]
            length {int} -- length of res_path
            inp {keras layer} -- input layer

        Returns:
            [keras layer] -- [output layer]
        '''

        shortcut = inp
        shortcut = self.conv2d_bn(shortcut, filters, 1, 1,
                                  activation=None, padding='same')

        out = self.conv2d_bn(inp, filters, 3, 3,
                             activation='relu', padding='same')

        out = keras.layers.add([shortcut, out])
        out = keras.layers.Activation('relu')(out)
        out = keras.layers.BatchNormalization(axis=3)(out)

        for _ in range(length-1):

            shortcut = out
            shortcut = self.conv2d_bn(shortcut, filters, 1, 1,
                                      activation=None, padding='same')

            out = self.conv2d_bn(out, filters, 3, 3,
                                 activation='relu', padding='same')

            out = keras.layers.add([shortcut, out])
            out = keras.layers.Activation('relu')(out)
            out = keras.layers.BatchNormalization(axis=3)(out)

        return out

    def create_model(self):
        '''
        Adds the output layer to the model with 'sigmoid' activation, compiles it and returns a keras model object.

        Returns:
        _model [keras model] -- keras model with MultiResUnet architecture
        '''

        inputs = keras.layers.Input(
            (self.input_height, self.input_width, self.input_channels))
        pad_input = keras.layers.ZeroPadding2D(self.padding)(inputs)

        mresblock1 = self.multi_res_block(self.n_filters, pad_input)
        pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock1)
        mresblock1 = self.res_path(self.n_filters, 4, mresblock1)

        mresblock2 = self.multi_res_block(self.n_filters*2, pool1)
        pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock2)
        mresblock2 = self.res_path(self.n_filters*2, 3, mresblock2)

        mresblock3 = self.multi_res_block(self.n_filters*4, pool2)
        pool3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock3)
        mresblock3 = self.res_path(self.n_filters*4, 2, mresblock3)

        mresblock4 = self.multi_res_block(self.n_filters*8, pool3)
        pool4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(mresblock4)
        mresblock4 = self.res_path(self.n_filters*8, 1, mresblock4)

        mresblock5 = self.multi_res_block(self.n_filters*16, pool4)

        up6 = keras.layers.concatenate([keras.layers.Conv2DTranspose(
            self.n_filters*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
        mresblock6 = self.multi_res_block(self.n_filters*8, up6)

        up7 = keras.layers.concatenate([keras.layers.Conv2DTranspose(
            self.n_filters*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
        mresblock7 = self.multi_res_block(self.n_filters*4, up7)

        up8 = keras.layers.concatenate([keras.layers.Conv2DTranspose(
            self.n_filters*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
        mresblock8 = self.multi_res_block(self.n_filters*2, up8)

        up9 = keras.layers.concatenate([keras.layers.Conv2DTranspose(self.n_filters, (2, 2), strides=(
            2, 2), padding='same')(mresblock8), mresblock1], axis=3)
        mresblock9 = self.multi_res_block(self.n_filters, up9)

        conv10 = self.conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')

        cropped = keras.layers.Cropping2D(self.padding)(conv10)
        _model = keras.Model(inputs=[inputs], outputs=[cropped])
        _model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.model = _model
        return self.model


# %% MAIN
if __name__ == "__main__":
    arch = UNET(64, (352, 296, 1), 'adam', .005, 'iou')
    model = arch.create_model()
    print(model.summary())
