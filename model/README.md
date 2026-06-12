# Model Assets

This directory is the local `microsoft/deberta-v3-large` base-model directory.
Training and inference load it directly with `from_pretrained("model")`.

Expected files include:

```text
config.json
pytorch_model.bin
spm.model
tokenizer_config.json
```

The original upstream model card is saved as `MODEL_CARD.md`. The directory is
intentionally flat because this project uses only one base model.
