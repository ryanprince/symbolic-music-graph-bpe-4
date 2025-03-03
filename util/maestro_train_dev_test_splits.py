from pathlib import Path
from random import shuffle, seed

train_proportion, dev_proportion, test_proportion = 0.8, 0.1, 0.1

patterns = ["**/*.mid", "**/*.midi"]
paths_for_pattern = lambda pattern: Path("./data/maestro-v3.0.0").resolve().glob(pattern)
midi_paths = [path for pattern in patterns for path in paths_for_pattern(pattern)]

bp_left, bp_right = int(len(midi_paths) * train_proportion), int(len(midi_paths) * (train_proportion + dev_proportion))

seed(777)
shuffle(midi_paths)

train_split = midi_paths[:bp_left]
dev_split = midi_paths[bp_left:bp_right]
test_split = midi_paths[bp_right:]