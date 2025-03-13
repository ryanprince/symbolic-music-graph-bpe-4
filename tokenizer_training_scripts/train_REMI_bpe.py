from miditok import REMI
from tokenizer_training_scripts.util.BaseTokenizerLearner import (
    BaseTokenizerLearner,
)
from util.maestro_train_dev_test_splits import train_split


def train_new_remi_bpe_tokenizer(vocab_size_target=1000, midi_file_paths=train_split):
    # Initialize the tokenizer.
    bpe_remi_tokenizer = REMI()

    # Train the tokenizer.
    bpe_remi_tokenizer.train(
        model="BPE",
        vocab_size=vocab_size_target,
        files_paths=midi_file_paths,
    )

    return bpe_remi_tokenizer


class RemiBpeLearner(BaseTokenizerLearner):
    def __init__(self):
        super().__init__(train_new_remi_bpe_tokenizer, "remi_bpe")


if __name__ == "__main__":
    learner = RemiBpeLearner()
    learner.train()
