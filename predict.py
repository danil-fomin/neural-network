import pickle
import sys
from pathlib import Path

import torch

from model import BowClassifier

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.pt"
VOCAB_PATH = ROOT / "vocab.pkl"

NUM_CLASSES = 3
IDX_TO_CLASS = {0: "negative", 1: "neutral", 2: "positive"}


def load():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    model = BowClassifier(len(vocab.token_to_idx), NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return vocab, model


def predict(text: str, vocab, model) -> tuple[str, list[float]]:
    bow = vocab.text_to_bow(text).unsqueeze(0)
    with torch.no_grad():
        logits = model(bow)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()
    pred_idx = max(range(NUM_CLASSES), key=lambda i: probs[i])
    return IDX_TO_CLASS[pred_idx], probs


def main() -> int:
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        print(f"ERROR: model or vocab not found in {ROOT}", file=sys.stderr)
        print("Run `python train.py` first.", file=sys.stderr)
        return 1

    vocab, model = load()

    print("Enter review text (empty line to exit):")
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        label, probs = predict(text, vocab, model)
        print(
            f"  {label}  (neg={probs[0]:.2f} neu={probs[1]:.2f} pos={probs[2]:.2f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
