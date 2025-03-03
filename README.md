## Scripts

`python ./fetch-maestro-dataset.py` -- Downloads and extracts the Maestro MIDI dataset into the `./data` directory. Overwrites the existing copy by erasing it first. After extracting its contents, moves the `.zip` file to the `./archive` directory, in case it's useful later.

`python ./generate-remi-tokenizer.py` -- Generates a REMI tokenizer which does not use BPE. Saves the result to a JSON file in the `./tokenizers directory` and adds a copy to the `./archive directory`.