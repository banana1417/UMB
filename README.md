# FakeNewsUMB reproduction

This folder contains a PyTorch/Transformers implementation adapted from the linked UMB repository and the attached paper.

Implemented components:

- three independent RoBERTa encoders for title, true reason, and false reason;
- ViT encoder for the image;
- CLS-to-token/patch bidirectional cross-modal interaction;
- Uncertainty Modeling Block (UMB) with diagonal Gaussian outputs;
- inverse-variance softmax fusion;
- diagonal-Gaussian 2-Wasserstein InfoNCE loss;
- reparameterized Monte Carlo classification loss;
- event-disjoint splitting, early stopping, and checkpoint saving;
- Qwen3-VL rationale-generation prompt and parser.

Expected record format (`json`, `jsonl`, or `csv`):

```json
{
  "image_path": "images/example.jpg",
  "title": "news title",
  "true_reason": "...",
  "false_reason": "...",
  "label": 0,
  "event_id": "event-001"
}
```

For reproducibility, the default hyperparameters follow the paper: batch size 16, maximum 20 epochs, `lambda_cont=0.10`, temperature `0.3`, `K=7`, and `H=8`. The model keeps all RoBERTa and ViT layers trainable unless `freeze_backbones=True` is set.

The training entry point is `train.py`; rationale generation is optional and is separated into `generate_rationales.py` so generated text can be cached before detector training.
