from MidiTok.src.miditok import REMI
from tokenizer_training_scripts.util.BaseTokenizerLearner import (
    BaseTokenizerLearner,
)


class RemiBasicLearner(BaseTokenizerLearner):
    def __init__(self):
        super().__init__(REMI, "remi_basic")


if __name__ == "__main__":
    learner = RemiBasicLearner()
    learner.train()
