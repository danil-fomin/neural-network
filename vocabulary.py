from collections import Counter


def tokenize(text: str):
    return text.lower().split()


class Vocabulary:
    def __init__(self, texts, max_size):
        self.token_to_idx = {}

        counter = Counter()

        for text in texts:
            tokens = tokenize(text)
            counter.update(tokens)

        top_tokens = counter.most_common(max_size)

        for index, (token, _) in enumerate(top_tokens):
            self.token_to_idx[token] = index + 1

    def __len__(self):
        return len(self.token_to_idx)

    def text_to_ids(self, text):
        ids = []
        tokens = tokenize(text)

        for token in tokens:
            if token in self.token_to_idx:
                ids.append(self.token_to_idx[token])

        return ids
