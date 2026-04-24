import pickle
import sys
from pathlib import Path

import torch

from model import RnnClassifier

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "model.pt"
VOCAB_PATH = ROOT / "vocab.pkl"

EMBED_DIM = 64
HIDDEN_SIZE = 64
MAX_LEN = 300
NUM_CLASSES = 3

IDX_TO_CLASS = {0: "negative", 1: "neutral", 2: "positive"}


def load():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RnnClassifier(len(vocab), EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    model.eval()
    return vocab, model


def predict(text: str, vocab, model) -> tuple[str, list[float]]:
    ids = vocab.text_to_ids(text)[:MAX_LEN]
    if not ids:
        ids = [vocab.token_to_idx["<UNK>"]]
    inputs = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(inputs)
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
