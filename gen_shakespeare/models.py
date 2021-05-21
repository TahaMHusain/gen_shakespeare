from typing import List, Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


class CustomModel(tf.keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int, rnn_units: int):
        super().__init__(self)
        self.embedding: Embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru: GRU = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense: Dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs: List[int], states: List[int] =None, return_state: bool = False, training: bool = False) \
            -> Tuple[List[int], List[int]]:
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

    def get_config(self):
        pass


class OneStep(tf.keras.Model):
    def __init__(self, model: Model, chars_from_ids: StringLookup, ids_from_chars: StringLookup,
                 temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "" or "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs: List[int], states: List[int] = None) -> Tuple[List[str], List[int]]:
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states

    def call(self, inputs, training=None, mask=None):
        pass

    def get_config(self):
        pass
