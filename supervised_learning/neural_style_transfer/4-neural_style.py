#!/usr/bin/env python3
"""
Update the class NST to calculate the style cost for a single layer:
"""
import numpy as np
import tensorflow as tf


class NST:
    """Update the class NST to calculate the style cost for a single layer:"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """initialize"""
        tf.enable_eager_execution()

        err_1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray):
            raise TypeError(err_1)
        if style_image.ndim != 3 or style_image.shape[-1] != 3:
            raise TypeError(err_1)
        err_2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(content_image, np.ndarray):
            raise TypeError(err_2)
        if content_image.ndim != 3 or content_image.shape[-1] != 3:
            raise TypeError(err_2)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """scale image"""
        err = "image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(image, np.ndarray):
            raise TypeError(err)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise TypeError(err)

        max_dim = 512
        long_dim = max(image.shape[:-1])
        scale = max_dim / long_dim
        new_shape = tuple(map(lambda x: int(scale * x), image.shape[:-1]))
        image = image[tf.newaxis, :]
        image = tf.image.resize_bicubic(image, new_shape)
        image = image / 255
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

        return image

    def load_model(self):
        """load model"""
        base_vgg = tf.keras.applications.VGG19(include_top=False,
                                               weights='imagenet',
                                               input_tensor=None,
                                               input_shape=None,
                                               pooling=None,
                                               classes=1000)
        custom_object = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        base_vgg.save('base_vgg')
        vgg = tf.keras.models.load_model('base_vgg',
                                         custom_objects=custom_object)

        for layer in vgg.layers:
            layer.trainable = False

        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """calculates grma matrix"""

        err = "input_layer must be a tensor of rank 4"
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError(err)
        if len(input_layer.shape) != 4:
            raise TypeError(err)
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        result = result / num_locations

        return result

    def generate_features(self):
        """extracts the style and content features"""

        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)

        outputs_style = self.model(style_image)
        style_outputs = outputs_style[:-1]

        outputs_content = self.model(content_image)
        content_ouput = outputs_content[-1]

        self.gram_style_features = [self.gram_matrix(style_output)
                                    for style_output in style_outputs]
        self.content_feature = content_ouput

    def layer_style_cost(self, style_output, gram_target):
        """calculates the style cost for a single
        style_output layer"""

        c = style_output.shape[-1]
        err_1 = "style_output must be a tensor of rank 4"
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError(err_1)
        if len(style_output.shape) != 4:
            raise TypeError(err_1)
        err_2 = ("gram_target must be a tensor of shape [1, {}, {}]".
                 format(c, c))
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(err_2)
        if gram_target.shape != (1, c, c):
            raise TypeError(err_2)

        gram_style = self.gram_matrix(style_output)
        style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return style_cost