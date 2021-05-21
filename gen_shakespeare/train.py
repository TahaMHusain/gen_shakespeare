import os
import time
import tensorflow as tf

from gen_shakespeare.data_processing import load_data, get_string_lookups, text_from_ids
from gen_shakespeare.models import CustomModel, OneStep

BATCH_SIZE = 64
BUFFER_SIZE = 10000
EPOCHS = 20
SEQ_LENGTH = 100

DATA_PATH = r'C:\Users\Mark\PycharmProjects\gen_shakespeare\data\\'
SPEAKER_CLASS = "FEMALE"
SPEAKER_PATH = SPEAKER_CLASS + '_cleaned.txt'
TRAIN_PATH = os.path.join(DATA_PATH, SPEAKER_PATH)
CKPT_DIR = r'C:\Users\Mark\PycharmProjects\gen_shakespeare\training_checkpoints\\'


# # NoneType error?
# def load_model():
#     return tf.saved_model.load(MODEL_PATH)


def get_latest_checkpoint(speaker_class: str):
    return tf.train.latest_checkpoint(os.path.join(CKPT_DIR, speaker_class))


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def get_dataset(text, ids_from_chars):
    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    sequences = ids_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)

    return (
        dataset.shuffle(BUFFER_SIZE)
               .batch(BATCH_SIZE, drop_remainder=True)
               .prefetch(tf.data.experimental.AUTOTUNE)
    )


def build_model(vocab_size):
    embedding_dim = 256
    rnn_units = 1024
    model = CustomModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units
    )

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss)
    return model, loss


def train_model(model, dataset):
    # Configure checkpoints
    checkpoint_prefix = os.path.join(CKPT_DIR, SPEAKER_CLASS, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )

    model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


def debug_model(model, dataset, loss, chars_from_ids, vocab):
    example_batch_predictions, input_example_batch, target_example_batch = [], [], []
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print("Input:\n", text_from_ids(input_example_batch[0], chars_from_ids).numpy())
    print()
    print("Next Char Predictions:\n", text_from_ids(sampled_indices, chars_from_ids).numpy())

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    mean_loss = example_batch_loss.numpy().mean()
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("Mean loss:        ", mean_loss)

    output_logits = tf.exp(mean_loss).numpy()
    print(f'{output_logits} should approximate {len(vocab)}')


def main():
    text, vocab = load_data(TRAIN_PATH)
    ids_from_chars, chars_from_ids = get_string_lookups(vocab)
    dataset = get_dataset(text, ids_from_chars)
    model, loss = build_model(len(ids_from_chars.get_vocabulary()))

    # debug_model(model, dataset, loss, chars_from_ids, vocab)

    train_model(model, dataset)


if __name__ == '__main__':
    main()
