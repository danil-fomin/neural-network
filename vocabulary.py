from collections import Counter
import re

def tokenize(text: str):
    lower_text = text.lower()
    clean_text = re.sub(r'[^\w\s]', '', lower_text)
    
    return clean_text.split()


class Vocabulary:
    def __init__(self, texts, max_size):
        self.token_to_idx = {}

        counter = Counter()

        for text in texts:
            tokens = tokenize(text)
            counter.update(tokens)

        top_tokens = counter.most_common(max_size)

        self.token_to_idx["<PAD>"] = 0
        self.token_to_idx["<UNK>"] = 1

        for token_id, (token, _) in enumerate(top_tokens, start=2):
            self.token_to_idx[token] = token_id

    def __len__(self):
        return len(self.token_to_idx)

    def text_to_ids(self, text):
        ids = []
        tokens = tokenize(text)

        unk_id = self.token_to_idx["<UNK>"]
        for token in tokens:
            ids.append(self.token_to_idx.get(token, unk_id))

        return ids
