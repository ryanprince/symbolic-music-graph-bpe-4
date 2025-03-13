## Getting Started

First, install any required dependencies.

Next, from the project directory, run the following.

 1. `python -m fetch-maestro-dataset` -- Fetch the Maestron dataset.
 2. `python -m tokenizer_training_scripts.train_REMI_basic` -- Learn the baseline tokenizer that uses REMI **without** BPE.
 3. `python -m tokenizer_training_scripts.train_REMI_bpe` -- Learn the baseline tokenizer that uses REMI **with** BPE.

## Directories

 * `trained_tokenizers` -- the output directory for the scripts in `tokenizer_training_scripts`

 * `tokenizer_training_scripts` -- learn tokenizers and output them into `trained_tokenizers`

 * `data` -- where the Maestro dataset gets downloaded

 * `util`

 * `archive` -- some timestamped versions of files will copy to here

## Tokenizer Learning Scripts

 * `python -m tokenizer_training_scripts.train_REMI_basic` -- Generates a REMI tokenizer that does not use BPE. Saves the result to a JSON file in the `./trained_tokenizers directory` and adds a copy to the `./archive directory`.

 * `python -m tokenizer_training_scripts.train_REMI_bpe` -- Generates a BPE tokenizer that's based on REMI tokens. Saves the result to a JSON file in the `./trained_tokenizers directory` and adds a copy to the `./archive directory`.

## Data Scripts

 * `python -m fetch-maestro-dataset` -- Downloads and extracts the Maestro MIDI dataset into a subdirectory within the `./data` directory. Overwrites the existing copy by erasing it first. After extracting its contents, moves the `.zip` file to the `./archive` directory, in case it's useful later.