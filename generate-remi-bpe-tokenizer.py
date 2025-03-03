from miditok import REMI

from util.filesystem import remove_tokenizer, save_tokenizer
from util.maestro_train_dev_test_splits import train_split

# The generated tokenizer will have a base vocabulary of REMI tokens plus additional BPE tokens that are learned.
# The vocab_size_target is the goal for the number of basic REMI tokens plus the number of learned BPE tokens.
# The number of REMI tokens in the vocabulary is constant. If vocab_size_target is less than the expected number
# of REMI tokens, then the method will not succeed.
def generate_bpe_remi_tokenizer(tokenizer_name, midi_file_paths, vocab_size_target):
    # Remove the tokenizer if it already exists.
    remove_tokenizer(tokenizer_name)

    # Initialize the tokenizer.
    bpe_remi_tokenizer = REMI()

    # Train the tokenizer.
    bpe_remi_tokenizer.train(
        model="BPE",
        vocab_size=vocab_size_target,
        files_paths=midi_file_paths,
    )

    # Save the tokenizer.
    save_tokenizer(bpe_remi_tokenizer, tokenizer_name)

    # Return the tokenizer.
    return bpe_remi_tokenizer

generate_bpe_remi_tokenizer('remi_bpe', train_split, 1000)
