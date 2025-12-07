import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_lengths(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [len(item.get("knowledge", "")) for item in data]


def plot_hist(lengths: List[int], title: str, out_path: Path, bin_size: int = 50) -> None:
    if not lengths:
        return
    max_len = max(lengths)
    bins = np.arange(0, max_len + bin_size, bin_size)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(lengths, bins=bins, color="#7396d8", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Length")
    ax.set_ylabel("Number")
    ax.set_title(title)
    ax.set_xticks(bins)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.set_xlim(0, bins[-1])
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    datasets = {
        "Korean": Path("data/korean_culture_train_200.json"),
        "English": Path("data/korean_culture_train_200_en.json"),
    }
    for name, path in datasets.items():
        lengths = load_lengths(path)
        out_path = path.with_name(f"{path.stem}_knowledge_hist.png")
        plot_hist(lengths, f"{name} knowledge length histogram", out_path)
        print(f"Saved {out_path} (bins: {len(np.arange(0, max(lengths) + 50, 50))})")


if __name__ == "__main__":
    main()
