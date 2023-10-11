from keras.layers import MultiHeadAttention, Dense, LayerNormalization,  Embedding, Layer, Add
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import pandas as pd

import numpy as np
import tensorflow as tf

class FeedForwardNetwork(Layer):
    def __init__(self, units):
        super().__init__()
        self.ffn = Sequential([
            Dense(units, activation='relu'),
            Dense(units, activation='relu')
        ])
        self.add = Add()
        self.norm = LayerNormalization()

    def __call__(self, x):
        x = self.add([x, self.ffn(x)])
        x = self.norm(x)
        return x



class GlobalSelfAttention(Layer):
    def __init__(self, num_head, units):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_head, key_dim=units)
        self.add = Add()
        self.norm = LayerNormalization()

    def __call__(self, x):
        attn_output, attn_score = self.mha(query=x, key=x, value=x, return_attention_scores=True)
        x = self.add([x, attn_output])
        x = self.norm(x)
        return x

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)

class EncoderLayer(Layer):
    def __init__(self, units, vocab_size, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mha = GlobalSelfAttention(num_head=num_heads, units=units)
        self.FFN = FeedForwardNetwork(units)
        self.add = Add()
        self.norm = LayerNormalization()

    def __call__(self, x):
        x = self.mha(x)

        x = self.FFN(x)

        return x

class Encoder(Layer):
    def __init__(self, units, vocab_size, d_model, num_heads, num_layer):
        super().__init__()
        self.d_model = d_model
        self.pe = positional_encoding(length=2048, depth=d_model)
        self.embedding = Embedding(vocab_size, units, mask_zero=True)
        self.enc_layers =[EncoderLayer(units, vocab_size, d_model, num_heads) for _ in range(num_layer)]

    def __call__(self, x):
        length = np.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pe[tf.newaxis, :length, :]

        for layer in self.enc_layers:
            x = layer(x)

        return x

class MaskedMultiHeadedAttention(Layer):
    def __init__(self, num_head, units):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_head, key_dim=units)
        self.add = Add()
        self.norm = LayerNormalization()

    def __call__(self, x):
        attention_output = self.mha(key=x, context=x, value=x, use_casual_mask=True)
        x = self.add([x, attention_output])
        x = self.norm(x)
        return x

class CrossAttention(Layer):
    def __init__(self, num_heads, units):
        super().__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=units)
        self.add = Add()
        self.norm = LayerNormalization()

    def __call__(self, x, context):
        attn__output = self.mha(query=x, key=context, value=context)
        x = self.add([x, attn__output])
        x = self.norm(x)

class DecoderLayer(Layer):
    def __init__(self, units, vocab_size, d_model, num_heads):
        super().__init__()
        self.mmha = MaskedMultiHeadedAttention(num_head=num_heads, units=units)
        self.ffn = FeedForwardNetwork(units=units)
        self.mha = CrossAttention(num_heads=num_heads, units=units)

    def __call__(self, x, context):
        x = self.mmha(x)
        x = self.mha(x, context)

        x = self.ffn(x)
        return x

class Decoder(Layer):
    def __init__(self, units, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.pe = positional_encoding(length=2048, depth=d_model)
        self.embedding = Embedding(vocab_size, units, mask_zero=True)

        self.dec_layers = [DecoderLayer(units=units, vocab_size=vocab_size, d_model=d_model, num_heads=num_heads) for _ in range(num_layers)]
        self.dense = Dense(vocab_size, activation='softmax')

    def __call__(self, x, context):
        length = np.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pe[tf.newaxis, :length, :]

        for layer in self.dec_layers:
            x = layer(x, context)

        x = self.dense(x)
        return x

class Transformer(Model):
    def __init__(self, units, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.encoder = Encoder(units=units, vocab_size=vocab_size, num_heads=num_heads, num_layer=num_layers, d_model=d_model)

        self.decoder = Decoder(units=units, vocab_size=vocab_size,
                               d_model=d_model, num_heads=num_heads,
                               num_layers=num_layers)

    def __call__(self, inputs):
        context, x = inputs

        context = self.encoder(context)

        x = self.decoder(x, context)

        return x


units = 1024
vocab_size = 5000
d_model = units
num_heads = 4
num_layers = 5

transformer = Transformer(units, vocab_size, d_model, num_heads, num_layers)
