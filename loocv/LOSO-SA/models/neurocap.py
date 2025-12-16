import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np

def squash(s, axis=-1):
    squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
    scale = squared_norm / (1 + squared_norm) / (tf.sqrt(squared_norm) + 1e-8)
    return scale * s

class AuditoryCaps(layers.Layer):
    def __init__(self, num_capsules, dim_capsules, routings=3):
        super(AuditoryCaps, self).__init__()
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routings = routings

    def build(self, input_shape):
        self.input_num_caps = input_shape[1]
        self.input_dim_caps = input_shape[2]
        self.W = self.add_weight(
            shape=[self.input_num_caps, self.num_capsules,
                   self.input_dim_caps, self.dim_capsules],
            initializer='glorot_uniform',
            name='W'
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        inputs = tf.expand_dims(inputs, 2)
        inputs = tf.tile(inputs, [1, 1, self.num_capsules, 1])
        u_hat = tf.einsum('bicd,icde->bice', inputs, self.W)
        b = tf.zeros(shape=[batch_size, self.input_num_caps, self.num_capsules])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=-1)
            c = tf.expand_dims(c, -1)
            s = tf.reduce_sum(c * u_hat, axis=1)
            v = squash(s)
            if i < self.routings - 1:
                agreement = tf.reduce_sum(u_hat * tf.expand_dims(v, 1), axis=-1)
                b += agreement
        return v

def margin_loss(y_true, y_pred):
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5
    L = y_true * tf.maximum(0., m_plus - y_pred)**2 + \
        lambda_ * (1 - y_true) * tf.maximum(0., y_pred - m_minus)**2
    return tf.reduce_mean(L)

def create_model(sample_len,channels_num, lr):
    num_classes = 10
    input_shape = (sample_len,channels_num)
    inputs = layers.Input(shape=input_shape)
    x = inputs

    self_att = TransformerBlock(embed_dim=channels_num, num_heads=4, ff_dim=32)(x)
    x = Channel_Attentioin()(self_att)
    
    conv1 = layers.Conv1D(128, 3, activation='relu', padding='same', strides=1, dilation_rate=1)(x)
    conv3 = layers.Conv1D(128, 3, activation='relu', padding='same', strides=1, dilation_rate=3)(conv1)
    conv9 = layers.Conv1D(128, 3, activation='relu', padding='same', strides=1, dilation_rate=9)(conv3)
    x = layers.Concatenate(axis=-1)([x, conv1, conv3, conv9])

    
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(256, 3, activation='relu', padding='same', strides=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, activation='linear', padding='same', strides=2)(x)
    
    x = layers.Reshape((-1, 16))(x)
    x = layers.Lambda(squash)(x)
    
    au_caps = AuditoryCaps(num_capsules=num_classes, dim_capsules=16, routings=3)(x)
    
    # Output layer
    outputs = tf.norm(au_caps, axis=-1)
    
    # Compile the model
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.legacy.Adam(learning_rate=lr),
        loss=margin_loss,
        metrics=['accuracy']
    )
    model.summary()
    return model



# Multi-Head Attention block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,  embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim), ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dropout2 = tf.keras.layers.Dropout(0.2)

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)  
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att' : self.att,
            'ffn' : self.ffn,
            'layernorm1' : self.layernorm1,
            'layernorm2' : self.layernorm2,
            'dropout1' : self.dropout1,
            'dropout2' : self.dropout2,
        })
        return config
    
class Channel_Attentioin(layers.Layer):
    def __init__(self, ratio=2, **kwargs):
        super(Channel_Attentioin, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.se_reduce = layers.Dense(input_shape[-1] // self.ratio, activation=tf.nn.leaky_relu)
        self.se_expand = layers.Dense(input_shape[-1], activation='sigmoid')
    
    def call(self, inputs):
        # Squeeze: Global Average Pooling
        squeeze = tf.reduce_mean(inputs, axis=-2, keepdims=True)
        
        # Excitation: Fully Connected layers
        excitation = self.se_reduce(squeeze)
        excitation = self.se_expand(excitation)
        
        # Scale
        scale = inputs * excitation
        return scale