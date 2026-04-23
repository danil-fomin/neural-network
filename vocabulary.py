from collections import Counter
import torch

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
            self.token_to_idx[token] = index

    def __len__(self):
        return len(self.token_to_idx)
    
    def text_to_ids(self, text):
        ids = []
        tokens = tokenize(text)

        for token in tokens:
            if token in self.token_to_idx:
                ids.append(self.token_to_idx[token])

        return ids
    
    def text_to_bow(self, text):
        ids = self.text_to_ids(text)
        vocab_size = len(self.token_to_idx)
        if not ids:
            return torch.zeros(vocab_size, dtype=torch.float32)
        ids_tensor = torch.tensor(ids, dtype=torch.long)
        
        return torch.bincount(ids_tensor, minlength=vocab_size).to(torch.float32)
