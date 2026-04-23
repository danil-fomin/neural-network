import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ReviewsDataset, BowReviewsDataset
from model import BowClassifier

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
MODEL_PATH = ROOT / "model.pt"
VOCAB_PATH = ROOT / "vocab.pkl"

NUM_CLASSES = 3
CLASS_NAMES = ["negative", "neutral", "positive"]
BATCH_SIZE = 64


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading vocab from {VOCAB_PATH}")
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    print(f"Loading model from {MODEL_PATH}")
    model = BowClassifier(len(vocab), NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, weights_only=True, map_location=device)
    )
    model.eval()

    print("Loading test dataset...")
    test_raw = ReviewsDataset(DATASET_DIR / "test.jsonl")
    print(f"  {len(test_raw)} samples")

    test_loader = DataLoader(
        BowReviewsDataset(test_raw, vocab),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    total = 0
    correct = 0
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            for true_label, pred_label in zip(labels.tolist(), predictions.tolist()):
                confusion[true_label][pred_label] += 1
                if true_label == pred_label:
                    correct += 1
                total += 1

    print()
    print(f"Total samples: {total}")
    print(f"Accuracy: {correct / total:.4f} ({correct}/{total})")
    print()

    print("Per-class accuracy:")
    for i, name in enumerate(CLASS_NAMES):
        class_total = sum(confusion[i])
        class_correct = confusion[i][i]
        if class_total:
            acc = class_correct / class_total
            print(f"  {name:10s}: {acc:.4f} ({class_correct}/{class_total})")
        else:
            print(f"  {name:10s}: no samples")
    print()

    print("Confusion matrix (rows = true, cols = predicted):")
    header = " " * 12 + "".join(f"{n:>10s}" for n in CLASS_NAMES)
    print(header)
    for i, name in enumerate(CLASS_NAMES):
        row = f"  {name:10s}" + "".join(
            f"{confusion[i][j]:>10d}" for j in range(NUM_CLASSES)
        )
        print(row)


if __name__ == "__main__":
    main()
