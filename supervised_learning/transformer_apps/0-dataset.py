#!/usr/bin/env python3
"""Module that loads and preps a dataset for machine translation."""
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and preps a dataset for machine translation."""

    def __init__(self):
        """Initializes the Dataset.

        Creates the instance attributes data_train, data_valid,
        tokenizer_pt, and tokenizer_en.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset.

        Args:
            data: tf.data.Dataset whose examples are formatted as a tuple
                (pt, en), where pt is the tf.Tensor containing the
                Portuguese sentence and en is the tf.Tensor containing the
                corresponding English sentence.

        Returns:
            Tuple of (tokenizer_pt, tokenizer_en).
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True
        )

        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_sentences, vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
