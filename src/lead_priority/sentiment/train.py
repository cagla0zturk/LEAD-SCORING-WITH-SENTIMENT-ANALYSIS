"""Fine-tune a multilingual transformer for engagement intent/sentiment.

Approach (matches the brief: "önceden eğitilmiş bir transformer + fine-tune, XLM-R / DistilBERT")
-----------------------------------------------------------------------------------------------
We fine-tune **``distilbert-base-multilingual-cased``** (configurable to ``xlm-roberta-base``)
on the synthetic, labelled interaction notes.

Why this model / why fine-tune (and not TF-IDF, not a giant LLM):

* **Multilingual TR + EN** is handled natively by the shared multilingual sub-word
  vocabulary - no language detection, no separate pipelines. The model has seen both
  Turkish and English during pre-training, so a "fiyatı yüksek buldu" and a "too expensive"
  land near each other in representation space.
* **Fine-tune > zero-shot here** because we have task-specific labels (our 4 CRM intents)
  and want a small, fast, self-contained serving artifact rather than a multi-GB LLM.
* **DistilBERT-multilingual (135M)** fine-tunes in ~3 min on CPU (no GPU on the box); XLM-R
  is a drop-in upgrade via ``SENTIMENT_BASE_MODEL`` when a GPU is available.

The training label is the *template class* that generated the text and is drawn
**independently of the tabular ``Converted`` target** - so there is no leakage from the
sentiment model into the conversion signal (see README).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from lead_priority.config import (
    RANDOM_SEED,
    SENTIMENT_BASE_MODEL,
    SENTIMENT_BATCH_SIZE,
    SENTIMENT_EPOCHS,
    SENTIMENT_LABELS,
    SENTIMENT_LEARNING_RATE,
    SENTIMENT_MAX_LENGTH,
    SENTIMENT_METRICS_PATH,
    SENTIMENT_MODEL_DIR,
    SENTIMENT_N_PER_CLASS,
    TEST_SIZE,
)
from lead_priority.data.synthetic_interactions import generate_interaction_dataset

logger = logging.getLogger(__name__)

LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}


@dataclass
class SentimentTrainResult:
    accuracy: float
    macro_f1: float
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: Path | None = None


class _TextDataset(Dataset):
    """Tokenised text classification dataset backed by pre-computed encodings."""

    def __init__(self, encodings: dict[str, torch.Tensor], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())
    return y_true, y_pred


def train_sentiment_model(
    *,
    n_per_class: int = SENTIMENT_N_PER_CLASS,
    epochs: int = SENTIMENT_EPOCHS,
    batch_size: int = SENTIMENT_BATCH_SIZE,
    max_length: int = SENTIMENT_MAX_LENGTH,
    base_model: str = SENTIMENT_BASE_MODEL,
    learning_rate: float = SENTIMENT_LEARNING_RATE,
    model_dir: Path = SENTIMENT_MODEL_DIR,
) -> SentimentTrainResult:
    """Fine-tune the transformer on synthetic notes, evaluate, and persist it."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    logger.info("Fine-tuning %s on %s", base_model, device)

    dataset = generate_interaction_dataset(n_per_class=n_per_class)
    texts = dataset.texts
    labels = [LABEL2ID[label] for label in dataset.labels]

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_SEED
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(SENTIMENT_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    def encode(batch_texts: list[str]) -> dict[str, torch.Tensor]:
        return dict(
            tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        )

    train_ds = _TextDataset(encode(x_train), y_train)
    test_ds = _TextDataset(encode(x_test), y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            labels_t = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch, labels=labels_t)
            out.loss.backward()
            optimizer.step()
            running_loss += float(out.loss.detach())
        logger.info("Epoch %d/%d - train loss %.4f", epoch, epochs, running_loss / len(train_loader))

    # --- Evaluate -------------------------------------------------------------------
    y_true, y_pred = _evaluate(model, test_loader, device)
    label_names = list(SENTIMENT_LABELS)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    accuracy = float(report["accuracy"])
    macro_f1 = float(report["macro avg"]["f1-score"])

    metrics: dict[str, Any] = {
        "model": base_model,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "labels": label_names,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "language_distribution": dataset.frame["language"].value_counts().to_dict(),
        "epochs": epochs,
        "max_length": max_length,
    }

    # --- Persist (HF directory format) ----------------------------------------------
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    SENTIMENT_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    logger.info(
        "Saved fine-tuned sentiment model to %s (acc=%.3f macro_f1=%.3f)",
        model_dir,
        accuracy,
        macro_f1,
    )

    return SentimentTrainResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        metrics=metrics,
        artifact_path=model_dir,
    )
