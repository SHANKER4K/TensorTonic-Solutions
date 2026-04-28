import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        for i,special in enumerate(["<PAD>","<UNK>","<BOS>","<EOS>"]):
            self.word_to_id[special] = i
            self.id_to_word[i] = special
            self.vocab_size += 1

        self.unique_words = sorted(set(word for text in texts for word in text.split()))
        for i,word in enumerate(self.unique_words):
            self.word_to_id[word.lower()] = i+4
            self.id_to_word[i+4] = word.lower()
            self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        encoded = sorted(text.lower().split())
        return [self.word_to_id[enc] if enc in self.unique_words else self.word_to_id['<UNK>'] for enc in encoded]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoded = ""
        print(ids)
        for id in ids:
            if self.id_to_word.get(id):
                decoded += self.id_to_word[id] + " "
            else:
                decoded += self.id_to_word[1] + " "
        return decoded.strip()
