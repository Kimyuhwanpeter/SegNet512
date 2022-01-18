# -*- coding:utf-8 -*-
from factory import Backbones
import tensorflow as tf

from keras import backend as K


class MaxPoolingWithArgmax2D(tf.keras.layers.Layer):

    def __init__(
            self,
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, pool_size[0], pool_size[1], 1]
        padding = padding.upper()
        strides = [1, strides[0], strides[1], 1]
        output, argmax = tf.nn.max_pool_with_argmax(
                inputs,
                ksize=ksize,
                strides=strides,
                padding=padding)
        argmax = tf.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
                dim//ratio[idx]
                if dim is not None else None
                for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        #with K.tf.variable_scope(self.name):
        with tf.compat.v1.variable_scope(self.name):
            mask = K.cast(mask, 'int32')
            input_shape = K.tf.shape(updates, out_type='int32')
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                        input_shape[0],
                        input_shape[1]*self.size[0],

                        input_shape[2]*self.size[1],
                        input_shape[3])
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype='int32')
            batch_shape = K.concatenate(
                    [[input_shape[0]], [1], [1], [1]],
                    axis=0)
            batch_range = tf.keras.backend.reshape(
                    tf.range(output_shape[0], dtype='int32'),
                    shape=batch_shape)
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype='int32')
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(K.reshape(
                K.stack([b, y, x, f]),
                [4, updates_size]))
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
                mask_shape[0],
                mask_shape[1]*self.size[0],
                mask_shape[2]*self.size[1],
                mask_shape[3]
                )

def UpSample2D(pool, indx, output_shape):

    #pool_ = tf.reshape(pool, [-1])
    pool_ = pool
    #batch_range = tf.reshape(tf.range(batch_size, dtype=indx.dtype), [tf.shape(pool)[0], 1, 1, 1])
    b = tf.expand_dims(tf.ones_like(indx), -1)
    #b = tf.reshape(b, [-1, 1])
    #indx_ = tf.reshape(indx, [-1, 1])
    indx_ = tf.expand_dims(indx, -1)
    indx_ = tf.concat([b, indx_], -1)
    ret = tf.scatter_nd(indx_, pool_, shape=[tf.shape(pool)[0], output_shape[1] * output_shape[2] * output_shape[3]])
    ret = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])

    return ret

def SegNet_model(input_shape=(512, 512, 3), classes=3, batch_size=4):

    h = inputs = tf.keras.Input(input_shape, batch_size=batch_size)    # per_image_standliazation ?? ???????
    
    backbone = Backbones.get_backbone(
        name="vgg16",
        input_shape=input_shape,
        weights="imagenet",
        include_top=False
    )
   
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv1")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv2")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    pool1, poo1_indx1 = MaxPoolingWithArgmax2D((2,2))(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="conv3")(pool1)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="conv4")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    pool2, poo1_indx2 = MaxPoolingWithArgmax2D((2,2))(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv5")(pool2)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv6")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv7")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    pool3, poo1_indx3 = MaxPoolingWithArgmax2D((2,2))(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv8")(pool3)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv9")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv10")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    pool4, poo1_indx4 = MaxPoolingWithArgmax2D((2,2))(h)

    ######################################################################################################

    h = MaxUnpooling2D((2,2))([pool4, poo1_indx4])
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv17")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv18")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv19")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = MaxUnpooling2D((2,2))([h, poo1_indx3])
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv20")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv21")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="conv22")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = MaxUnpooling2D((2,2))([h, poo1_indx2])
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="conv23")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv24")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = MaxUnpooling2D((2,2))([h, poo1_indx1])
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv25")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv26")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=classes, kernel_size=1, padding="same", name="outputs")(h)

    SegModel = tf.keras.Model(inputs=inputs, outputs=h)

    SegModel.get_layer("conv1").set_weights(backbone.get_layer("block1_conv1").get_weights())
    SegModel.get_layer("conv2").set_weights(backbone.get_layer("block1_conv2").get_weights())
    SegModel.get_layer("conv3").set_weights(backbone.get_layer("block2_conv1").get_weights())
    SegModel.get_layer("conv4").set_weights(backbone.get_layer("block2_conv2").get_weights())
    SegModel.get_layer("conv5").set_weights(backbone.get_layer("block3_conv1").get_weights())
    SegModel.get_layer("conv6").set_weights(backbone.get_layer("block3_conv2").get_weights())
    SegModel.get_layer("conv7").set_weights(backbone.get_layer("block3_conv3").get_weights())
    SegModel.get_layer("conv8").set_weights(backbone.get_layer("block4_conv1").get_weights())
    SegModel.get_layer("conv9").set_weights(backbone.get_layer("block4_conv2").get_weights())
    SegModel.get_layer("conv10").set_weights(backbone.get_layer("block4_conv3").get_weights())

    return SegModel