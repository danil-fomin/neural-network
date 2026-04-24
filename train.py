import pickle
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ReviewsDataset, SequenceReviewsDataset, collate_pad
from vocabulary import Vocabulary
from model import RnnClassifier

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
MODEL_PATH = ROOT / "model.pt"
VOCAB_PATH = ROOT / "vocab.pkl"

VOCAB_SIZE = 10000
EMBED_DIM = 64
HIDDEN_SIZE = 64
MAX_LEN = 300
NUM_CLASSES = 3
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 10
NUM_WORKERS = 2


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total += inputs.size(0)
    return total_loss / total, total_correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading train dataset...")
    t0 = time.time()
    train_raw = ReviewsDataset(DATASET_DIR / "train.jsonl")
    print(f"  {len(train_raw)} samples in {time.time() - t0:.1f}s")

    print("Loading val dataset...")
    t0 = time.time()
    val_raw = ReviewsDataset(DATASET_DIR / "val.jsonl")
    print(f"  {len(val_raw)} samples in {time.time() - t0:.1f}s")

    print("Building vocabulary...")
    t0 = time.time()
    train_texts = [text for text, _ in train_raw.samples]
    vocab = Vocabulary(train_texts, max_size=VOCAB_SIZE)
    print(f"  vocab size: {len(vocab)} in {time.time() - t0:.1f}s")

    train_loader = DataLoader(
        SequenceReviewsDataset(train_raw, vocab, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_pad,
    )

    val_loader = DataLoader(
        SequenceReviewsDataset(val_raw, vocab, MAX_LEN),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_pad,
    )

    model = RnnClassifier(len(vocab), EMBED_DIM, HIDDEN_SIZE, NUM_CLASSES).to(device)
    print(f"Model: {model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    label_counts = [0] * NUM_CLASSES
    for _, label in train_raw.samples:
        label_counts[label] += 1
    total = sum(label_counts)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * c) for c in label_counts],
        dtype=torch.float32,
    ).to(device)
    
    print(f"Label counts: {label_counts}")
    print(f"Class weights: {[round(w, 3) for w in class_weights.tolist()]}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Saving vocab to {VOCAB_PATH}")
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)

    best_val_acc = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1)
            epoch_correct += (predictions == labels).sum().item()
            epoch_total += inputs.size(0)

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        elapsed = time.time() - t0
        marker = ""
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            marker = " *saved*"
        print(
            f"Epoch {epoch}/{EPOCHS}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

    print(f"Best val_acc: {best_val_acc:.4f} (saved to {MODEL_PATH})")
    print("Done.")


if __name__ == "__main__":
    main()
