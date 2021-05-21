import time
import os

import tensorflow as tf

from gen_shakespeare.data_processing import load_data, get_string_lookups
from gen_shakespeare.train import build_model, get_latest_checkpoint
from gen_shakespeare.models import CustomModel, OneStep

DATA_PATH = r'C:\Users\Mark\PycharmProjects\gen_shakespeare\data\\'
SPEAKER_CLASS = "FEMALE"
SPEAKER_PATH = SPEAKER_CLASS + '_cleaned.txt'
TEST_PATH = os.path.join(DATA_PATH, SPEAKER_PATH)
INIT_STR = {"MALE": "ROMEO:", "FEMALE": "JULIET", "HIGHSTATUS": "HAMLET:", "LOWSTATUS": "TOWNSMAN:"}
GEN_LENGTH = 1000


def load_model() -> OneStep:
    latest = get_latest_checkpoint(SPEAKER_CLASS)
    text, vocab = load_data(TEST_PATH)
    ids_from_chars, chars_from_ids = get_string_lookups(vocab)

    model, t = build_model(len(ids_from_chars.get_vocabulary()))
    model.load_weights(latest)
    return OneStep(model, chars_from_ids, ids_from_chars)


def generate_text(one_step_model: OneStep):
    start = time.time()
    states = None
    next_char = tf.constant([INIT_STR[SPEAKER_CLASS]])
    result = [next_char]

    for n in range(GEN_LENGTH):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('UTF-8'), '\n\n' + '_' * 80)
    print('\nRun time:', end - start)


def main():
    one_step_model = load_model()
    generate_text(one_step_model)


if __name__ == "__main__":
    main()
