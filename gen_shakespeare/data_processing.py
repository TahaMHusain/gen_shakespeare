from typing import Tuple, List, Set

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

DEBUG_PATH = r'C:\Users\Mark\PycharmProjects\gen_shakespeare\data\MALE_cleaned.txt'


def text_from_ids(ids: List[int], chars_from_ids) -> str:
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


def load_data(path: str) -> Tuple[str, Set[str]]:
    with open(path, 'rb') as file:
        text = file.read().decode('UTF-8')
    vocab = sorted(set(text))
    print(len(vocab))
    return text, vocab


def get_string_lookups(vocab: Set[str]) -> Tuple[StringLookup, StringLookup]:
    ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
    chars_from_ids = preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)
    return ids_from_chars, chars_from_ids


def debug_data(ids_from_chars: StringLookup) -> None:
    example_text = 'abcde'
    chars = tf.strings.unicode_split(example_text, input_encoding='UTF-8')
    print(chars)
    ids = ids_from_chars(chars)
    print(ids)


def main():
    text, vocab = load_data(DEBUG_PATH)
    ids_from_chars, chars_from_ids = get_string_lookups(vocab)
    debug_data(vocab, ids_from_chars, chars_from_ids)


if __name__ == '__main__':
    main()
