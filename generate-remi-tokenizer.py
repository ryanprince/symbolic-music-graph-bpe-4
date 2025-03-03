from miditok import REMI
from util.filesystem import remove_tokenizer, save_tokenizer

tokenizer_name = 'remi_basic'

remove_tokenizer(tokenizer_name)

learn_remi_tokenization = REMI()
save_tokenizer(learn_remi_tokenization, tokenizer_name)