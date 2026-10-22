"""Character spelling is retained; protocol markers/placeholders are atomic tokens.

An ID selects one embedding row. It is not a word meaning or an embedding.
The alphabet is fixed in advance, not fitted to validation/test examples.
"""
import hashlib
import json
import string

CONTROL = ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<INPUT>", "<OUTPUT>")
PLACEHOLDERS = ("[NAME]", "[DOCTOR]", "[DATE]", "[MRN]", "[PHONE]", "[ADDRESS]")


class CharacterTokenizer:
    def __init__(self):
        self.tokens = list(CONTROL + PLACEHOLDERS) + list(string.ascii_letters + string.digits + string.punctuation + " \n\t")
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.atomic = sorted(CONTROL + PLACEHOLDERS, key=len, reverse=True)
        self.pad_id = self.token_to_id["<PAD>"]
        self.eos_id = self.token_to_id["<EOS>"]

    def __len__(self):
        return len(self.tokens)

    def encode_with_offsets(self, text):
        """Return IDs and half-open character spans [start, end) into this text."""
        ids, offsets, cursor = [], [], 0
        while cursor < len(text):
            token = next((marker for marker in self.atomic if text.startswith(marker, cursor)), text[cursor])
            if token not in self.token_to_id:
                raise ValueError(f"Unsupported character {token!r} at offset {cursor}; no silent <UNK> replacement")
            ids.append(self.token_to_id[token])
            offsets.append((cursor, cursor + len(token)))
            cursor += len(token)
        return ids, offsets

    def encode(self, text):
        return self.encode_with_offsets(text)[0]

    def decode(self, ids):
        if any(index < 0 or index >= len(self) for index in ids):
            raise ValueError("ID outside tokenizer vocabulary")
        return "".join(self.tokens[index] for index in ids)

    def metadata(self):
        return {"type": "ascii_character_atomic_v1", "tokens": self.tokens, "unsupported": "error"}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.metadata(), sort_keys=True).encode()).hexdigest()

    def prompt(self, source):
        if any(marker in source for marker in CONTROL):
            raise ValueError("Source must not contain reserved protocol markers")
        return "<BOS><INPUT>" + source + "<OUTPUT>"
