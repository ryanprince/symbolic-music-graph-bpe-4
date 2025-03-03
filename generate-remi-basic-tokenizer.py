from miditok import REMI

from util.filesystem import remove_tokenizer, save_tokenizer
from util.maestro_train_dev_test_splits import train_split

# Learning REMI tokens is a sort of trivial case when BPE is not involved.
# The output is the same regardless of the training data; no training data is needed.
# The number of REMI tokens in the vocabulary is constant.
def generate_remi_tokenizer(tokenizer_name):
    # Remove the tokenizer if it already exists.
    remove_tokenizer(tokenizer_name)

    # Initialize the tokenizer.
    remi_tokenizer = REMI()

    # No training to do.

    # Save the tokenizer.
    save_tokenizer(remi_tokenizer, tokenizer_name)

    # Return the tokenizer.
    return remi_tokenizer

generate_remi_tokenizer('remi_bpe', train_split)