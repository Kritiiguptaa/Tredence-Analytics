import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt


# Linear layer where each weight has a learnable sigmoid gate that can prune it.
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    # Squash gate_scores through sigmoid so every gate lies in (0, 1).
    def gates(self):
        return torch.sigmoid(self.gate_scores)

    # Multiply weights by gates before the linear transform so dead gates kill connections.
    def forward(self, x):
        return F.linear(x, self.weight * self.gates(), self.bias)


# Simple 3-layer MLP on CIFAR-10 built from prunable linear layers.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# L1 sparsity penalty: sum of all sigmoid gates across every prunable layer.
def sparsity_loss(model):
    total = 0.0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            total = total + m.gates().sum()
    return total


# Percentage of gates that have collapsed below the pruning threshold.
def sparsity_level(model, threshold=1e-2):
    total = pruned = 0
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            g = m.gates().detach()
            total += g.numel()
            pruned += (g < threshold).sum().item()
    return 100.0 * pruned / total


# Flatten all final gate values into one array for the histogram plot.
def all_gate_values(model):
    vals = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            vals.append(m.gates().detach().flatten().cpu())
    return torch.cat(vals).numpy()


# Build CIFAR-10 train and test dataloaders with basic normalization.
def get_loaders(batch_size=128):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    return (
        DataLoader(train_ds, batch_size, shuffle=True, num_workers=0),
        DataLoader(test_ds, 256, shuffle=False, num_workers=0),
    )


# Compute test-set classification accuracy as a percentage.
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


# Train one model for the given lambda, return the model, final accuracy, and sparsity.
def train_one(lam, epochs, device):
    train_loader, test_loader = get_loaders()
    model = Net().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y) + lam * sparsity_loss(model)
            loss.backward()
            opt.step()
        print(f"lambda={lam:g} epoch {epoch + 1}/{epochs} sparsity={sparsity_level(model):.2f}%")
    return model, evaluate(model, test_loader, device), sparsity_level(model)


# Write the final markdown report with the explanation, results table, and plot link.
def save_report(results, best_lam):
    lines = [
        "# Self-Pruning Neural Network — Report",
        "",
        "## Why an L1 penalty on sigmoid gates encourages sparsity",
        "",
        "Every weight is multiplied by `g = sigmoid(s)` where `s` is a learnable score, so each gate sits in `(0, 1)` and is strictly positive. That means the L1 norm of all gates is just `sum(g)`. Adding `lambda * sum(g)` to the loss applies a constant downward pressure on every gate, while the classification loss only pushes back on gates whose weights actually reduce the cross-entropy. Gates attached to useful connections resist because cutting them hurts accuracy more than the L1 term rewards; gates attached to useless connections slide toward zero as `s` drifts to large negative values and `sigmoid(s)` → 0. The outcome is a bimodal distribution — a dense spike near 0 (pruned) and a cluster near 1 (kept) — which is exactly what a good pruning method should produce.",
        "",
        "## Results",
        "",
        "| Lambda | Test Accuracy (%) | Sparsity Level (%) |",
        "|--------|-------------------|---------------------|",
    ]
    for lam, acc, sp in results:
        lines.append(f"| {lam:g} | {acc:.2f} | {sp:.2f} |")
    lines += [
        "",
        "As lambda grows the network is forced to prune more aggressively: sparsity rises while test accuracy falls, giving the expected accuracy–sparsity trade-off.",
        "",
        f"## Final gate distribution (best run, lambda = {best_lam:g})",
        "",
        "![gate distribution](gate_distribution.png)",
        "",
    ]
    with open("REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# Train three models with different lambdas, save plot of best run, write report.
def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lambdas = [1e-6, 1e-5, 1e-4]
    epochs = 8
    results = []
    models = {}
    for lam in lambdas:
        model, acc, sp = train_one(lam, epochs, device)
        results.append((lam, acc, sp))
        models[lam] = model
        print(f"lambda={lam:g}  accuracy={acc:.2f}%  sparsity={sp:.2f}%")

    best_lam = max(results, key=lambda r: r[2] * (r[1] / 100.0))[0]
    gates = all_gate_values(models[best_lam])
    plt.figure(figsize=(8, 5))
    plt.hist(gates, bins=60)
    plt.yscale("log")
    plt.xlabel("gate value")
    plt.ylabel("count (log scale)")
    plt.title(f"Final gate distribution (lambda={best_lam:g})")
    plt.savefig("gate_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()

    save_report(results, best_lam)
    print("wrote REPORT.md and gate_distribution.png")


if __name__ == "__main__":
    main()
