import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from config import UMBConfig
from data import UMBFakeNewsDataset, event_disjoint_split, load_records, seed_everything
from losses import total_loss
from model import FakeNewsUMB


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def build_splits(records, seed):
    split_values = {record.get("split") for record in records}
    if split_values >= {"train", "val", "test"}:
        return (
            [r for r in records if r.get("split") == "train"],
            [r for r in records if r.get("split") == "val"],
            [r for r in records if r.get("split") == "test"],
        )
    return event_disjoint_split(records, seed=seed)


def evaluate(model, loader, device, config):
    model.eval()
    labels, predictions = [], []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            target = batch.pop("label")
            outputs = model(**batch)
            loss_dict = total_loss(
                outputs,
                target,
                contrastive_weight=config.contrastive_weight,
                temperature=config.contrastive_temperature,
                samples=config.reparameterization_samples,
                eps=config.eps,
            )
            losses.append(loss_dict["loss"].item())
            labels.extend(target.long().cpu().tolist())
            predictions.extend((outputs["prob"] >= 0.5).long().cpu().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": accuracy_score(labels, predictions) if labels else float("nan"),
        "f1": f1_score(labels, predictions, zero_division=0) if labels else float("nan"),
    }


def train_one_epoch(model, loader, optimizer, scaler, device, config):
    model.train()
    running = []
    use_amp = scaler is not None
    for batch in loader:
        batch = move_batch(batch, device)
        labels = batch.pop("label")
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(**batch)
                loss_dict = total_loss(
                    outputs, labels,
                    contrastive_weight=config.contrastive_weight,
                    temperature=config.contrastive_temperature,
                    samples=config.reparameterization_samples,
                    eps=config.eps,
                )
            scaler.scale(loss_dict["loss"]).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            loss_dict = total_loss(
                outputs, labels,
                contrastive_weight=config.contrastive_weight,
                temperature=config.contrastive_temperature,
                samples=config.reparameterization_samples,
                eps=config.eps,
            )
            loss_dict["loss"].backward()
            optimizer.step()
        running.append(loss_dict["loss"].item())
    return float(np.mean(running)) if running else float("nan")


def main(args):
    config = UMBConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
    )
    seed_everything(config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    records = load_records(args.records, args.image_root)
    train_records, val_records, test_records = build_splits(records, config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(config.vision_model_name)

    train_set = UMBFakeNewsDataset(train_records, tokenizer, image_processor, config.max_text_length)
    val_set = UMBFakeNewsDataset(val_records, tokenizer, image_processor, config.max_text_length)
    test_set = UMBFakeNewsDataset(test_records, tokenizer, image_processor, config.max_text_length)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

    model = FakeNewsUMB(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, config)
        val_metrics = evaluate(model, val_loader, device, config)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "config": config.__dict__, "epoch": epoch}, output_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device, config)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(json.dumps({"test": test_metrics}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output-dir", default="runs/umb")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    main(parser.parse_args())
