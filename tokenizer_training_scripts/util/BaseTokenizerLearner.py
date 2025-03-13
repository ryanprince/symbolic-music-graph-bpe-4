import os
from util.filesystem import copy_file_to_archive


# Default tokenizer settings
DEFAULT_TOKENIZERS_DIR = os.path.abspath("./trained_tokenizers")


def tokenizer_path(tokenizers_directory, name, extension=".json"):
    return f"{tokenizers_directory}/{name}{extension}"


def remove_tokenizer(tokenizers_directory, name, extension=".json"):
    file_path = tokenizer_path(tokenizers_directory, name, extension)
    if os.path.exists(file_path):
        os.remove(file_path)


def save_tokenizer(tokenizers_directory, tokenizer, name, extension=".json"):
    os.makedirs(tokenizers_directory, exist_ok=True)
    file_path = tokenizer_path(tokenizers_directory, name, extension)
    tokenizer.save(file_path)
    copy_file_to_archive(file_path)


class BaseTokenizerLearner:
    """
    A pipeline for training and saving a tokenizer.
    Attributes:
      train_new_tokenizer (function): A function that initializes a tokenizer, trains the tokenizer as appropriate, and then returns the trained tokenizer.
      tokenizer_name (str): The name of the tokenizer.
      tokenizers_directory (str): The directory where the trained tokenizer should be saved.
      archive_directory (str): The directory where archives are stored.
    Methods:
      train(): Removes the existing tokenizer from tokenizers_directory if it exists, trains a new tokenizer, saves it, and returns the trained tokenizer.
    """

    def __init__(
        self,
        train_new_tokenizer,
        tokenizer_name,
        tokenizers_directory=DEFAULT_TOKENIZERS_DIR,
    ):
        self.train_new_tokenizer = train_new_tokenizer
        self.tokenizer_name = tokenizer_name
        self.tokenizers_directory = tokenizers_directory

    def train(self):
        # Remove the tokenizer if it already exists
        remove_tokenizer(self.tokenizers_directory, self.tokenizer_name)

        # Train the tokenizer
        tokenizer = self.train_new_tokenizer()

        # Save the tokenizer
        save_tokenizer(
            self.tokenizers_directory,
            tokenizer,
            self.tokenizer_name,
        )

        # Return the tokenizer
        return tokenizer
