#!/usr/bin/env python3
"""Update the class NST to compute the gradients for the generated image:"""

import numpy as np
import tensorflow as tf


class NST:
    """Update the class NST to compute the gradients for the generated image:
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """initialize"""

        if type(style_image) is not np.ndarray \
                or len(style_image.shape) != 3 \
                or style_image.shape[2] != 3:
            msg = 'style_image must be a numpy.ndarray with shape (h, w, 3)'
            raise TypeError(msg)

        if type(content_image) is not np.ndarray \
                or len(content_image.shape) != 3 \
                or content_image.shape[2] != 3:
            msg = 'content_image must be a numpy.ndarray with shape (h, w, 3)'
            raise TypeError(msg)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            msg = 'alpha must be a non-negative number'
            raise TypeError(msg)

        if not isinstance(beta, (int, float)) or beta < 0:
            msg = 'beta must be a non-negative number'
            raise TypeError(msg)
        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """scale image"""
        if type(image) is not np.ndarray \
                or len(image.shape) != 3 \
                or image.shape[2] != 3:
            msg = 'image must be a numpy.ndarray with shape (h, w, 3)'
            raise TypeError(msg)

        h, w, c = image.shape
        if w > h:
            w_new = 512
            h_new = int(h * 512 / w)
        else:
            h_new = 512
            w_new = int(w * 512 / h)

        dim = (h_new, w_new)
        image = image[tf.newaxis, ...]
        image = tf.image.resize_bicubic(image, dim, align_corners=False)
        image = tf.math.divide(image, 255)
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

        return image

    def load_model(self):
        """ loads the model for neural style transfer """
        vgg_pre = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                    weights='imagenet')
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg_pre.save("base_model")
        vgg = tf.keras.models.load_model("base_model",
                                         custom_objects=custom_objects)
        for layer in vgg.layers:
            layer.trainable = False
        style_outputs = \
            [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = vgg.get_layer(self.content_layer).output
        model_outputs = style_outputs + [content_outputs]
        self.model = tf.keras.models.Model(vgg.input, model_outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """calculate gram matrix"""
        e = 'input_layer must be a tensor of rank 4'
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) \
                or len(input_layer.shape) != 4:
            raise TypeError(e)
        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        """calculate neural style cost"""

        vgg19 = tf.keras.applications.vgg19
        content_image_input = vgg19.preprocess_input(self.content_image * 255)
        style_image_input = vgg19.preprocess_input(self.style_image * 255)
        content_img_output = self.model(content_image_input)
        style_img_output = self.model(style_image_input)
        list_gram = []
        for out in style_img_output[:-1]:
            list_gram = list_gram + [self.gram_matrix(out)]
        self.gram_style_features = list_gram
        self.content_feature = content_img_output[-1]

    def layer_style_cost(self, style_output, gram_target):
        """layer style cost"""
        err = 'style_output must be a tensor of rank 4'
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError(err)

        c = int(style_output.shape[-1])
        err = 'gram_target must be a tensor of shape [1, {}, {}]'.format(c, c)
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(err)

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """style cost"""
        my_length = len(self.style_layers)
        err = \
            'style_outputs must be a list with a length of {}'. \
            format(my_length)
        if (not type(style_outputs) is list
                or len(self.style_layers) != len(style_outputs)):
            raise TypeError(err)
        weight_per_style_layer = 1.0 / float(my_length)
        style_cost = 0
        for target_style, comb_style in \
                zip(self.gram_style_features, style_outputs):
            style_cost += \
                weight_per_style_layer * \
                self.layer_style_cost(comb_style, target_style)

        return style_cost

    def content_cost(self, content_output):
        """Content cost"""
        s = self.content_feature.shape
        err = 'content_output must be a tensor of shape {}'.format(s)
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(err)
        if self.content_feature.shape != content_output.shape:
            raise TypeError(err)
        if len(content_output.shape) == 3:
            content_output = tf.expand_dims(content_output, 0)
        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """calculate the total cost"""
        s = self.content_image.shape
        m = "generated_image must be a tensor of shape {}".format(s)
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(m)
        if generated_image.shape != s:
            raise TypeError(m)
        vgg19 = tf.keras.applications.vgg19
        generated_input = vgg19.preprocess_input(generated_image * 255)
        generated_output = self.model(generated_input)
        J_style = self.style_cost(generated_output[:-1])
        J_content = self.content_cost(generated_output[-1])
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """compute the gradients"""
        s = self.content_image.shape
        err = "generated_image must be a tensor of shape {}".format(s)
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(err)
        if generated_image.shape != s:
            raise TypeError(err)

        with tf.GradientTape() as tape:
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradient = tape.gradient(J_total, generated_image)
        return gradient, J_total, J_content, J_style
