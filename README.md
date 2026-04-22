# Self-Pruning Neural Network

A feed-forward network on CIFAR-10 whose weights each carry a learnable sigmoid gate. An L1 penalty on the gates drives most of them to zero during training, pruning the network on the fly.

## Run

```
pip install -r requirements.txt
python train.py
```

This trains the model three times (lambda = 1e-6, 1e-5, 1e-4), prints per-epoch sparsity, writes `REPORT.md` with the results table and explanation, and saves `gate_distribution.png`.

## Files

- `train.py` — `PrunableLinear` layer, the network, training loop, evaluation, plot, and report writer.
- `REPORT.md` — generated after training.
- `gate_distribution.png` — generated after training.
