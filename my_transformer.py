import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential


class MLP(layers.Layer):
    def __init__(self, hidden_units, dropout):
        super().__init__()
        self.linear = layers.Dense(hidden_units, activation="gelu")
        self.dropout = layers.Dropout(dropout)

    def call(self, x):
        out = self.linear(x)
        out = self.dropout(out)
        return out


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


class Transformer(layers.Layer):
    def __init__(self, num_heads, proj_dim, dropout, hidden_units):
        super().__init__()
        self.dropout = dropout
        self.hidden_units = hidden_units
        self.layerNorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.multiHeadAttention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=proj_dim, dropout=self.dropout,
            kernel_regularizer=tf.keras.regularizers.L2(0.0001)
        )
        self.layerNorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(hidden_units, dropout)

    def call(self, x):
        x1 = self.layerNorm1(x)
        attention_output = self.multiHeadAttention(x1, x1)
        # skip connection
        x2 = attention_output + x
        x3 = self.layerNorm2(x2)
        x3 = self.mlp(x3)
        # skip connection
        out = x3 + x2
        return out


def build_vit(patch_size, num_patches, proj_dim, num_transformer, num_heads):
    model = Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))
    model.add(Patches(patch_size))
    model.add(PatchEncoder(num_patches, proj_dim))
    for i in range(num_transformer):
        model.add(Transformer(num_heads=num_heads, proj_dim=proj_dim, dropout=0.1, hidden_units=proj_dim))
    return model


def build_vit_with_classifier(model_vit, num_classes, reg=False, lamb=0.0):
    model = Sequential()
    for layer in model_vit.layers:
        model.add(layer)
    model.add(layers.LayerNormalization(epsilon=1e-6))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    if reg:
        model.add(layers.Dense(1024, activation="gelu", kernel_regularizer=tf.keras.regularizers.L2(lamb)))
    else:
        model.add(layers.Dense(1024, activation="gelu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))
    return model
