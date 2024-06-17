#!/usr/bin/env python3
"""
Defines a Transformer model for machine translation.
"""

import tensorflow as tf
from 4-positional_encoding import positional_encoding
from 7-transformer_encoder_block import EncoderBlock
from 8-transformer_decoder_block import DecoderBlock

class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for i, block in enumerate(self.blocks):
            x, block1, block2 = block(x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab, max_seq_input, max_seq_target, drop_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask, decoder_mask):
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output, _ = self.decoder(target, enc_output, training, look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)
        return final_output

