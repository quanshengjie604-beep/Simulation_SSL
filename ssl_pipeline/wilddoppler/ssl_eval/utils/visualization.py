from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    from umap import UMAP
except ImportError:  # pragma: no cover - optional dependency
    UMAP = None

from .feature_utils import FeaturePack, gather_labels, subsample_pack


def reduce_dimensionality(
    features: np.ndarray,
    method: str,
    perplexity: float,
    seed: int,
) -> np.ndarray:
    n_samples = features.shape[0]
    if n_samples < 2:
        raise ValueError("Dimensionality reduction requires at least two samples.")
    method = str(method).lower()
    if method == "pca":
        reducer = PCA(n_components=3, random_state=seed)
        transformed = reducer.fit_transform(features)
        return transformed[:, 1:3]

    if method == "umap":
        if UMAP is None:
            raise ImportError("umap-learn is not installed. Install it to use the UMAP projection.")
        n_neighbors = min(n_samples, max(5, n_samples // 10))
        n_neighbors = min(30, max(2, n_neighbors))
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        )
        return reducer.fit_transform(features)

    max_valid = max(5, n_samples - 1)
    tuned_perplexity = min(perplexity, max_valid)
    if tuned_perplexity >= n_samples:
        tuned_perplexity = max(5, n_samples // 2)

    reducer = TSNE(
        n_components=2,
        init="random",
        perplexity=tuned_perplexity,
        random_state=seed,
    )
    return reducer.fit_transform(features)


def plot_embedding(
    coords: np.ndarray,
    labels: Sequence[str],
    title: str,
    save_path: Path | None,
    marker_size: float = 15.0,
) -> None:
    plt.figure(figsize=(8, 8))
    labels_array = np.array(labels)
    unique_labels = np.unique(labels_array)

    def _build_color_lookup(entries: Sequence[str]) -> Dict[str, tuple]:
        n_classes = len(entries)
        if n_classes <= 10:
            cmap = plt.get_cmap("tab10", n_classes)
        elif n_classes <= 20:
            cmap = plt.get_cmap("tab20", n_classes)
        else:
            cmap = plt.get_cmap("gist_ncar", n_classes)
        return {label: cmap(idx) for idx, label in enumerate(entries)}

    color_lookup = _build_color_lookup(unique_labels)

    for lab in unique_labels:
        mask = labels_array == lab
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=str(lab),
            alpha=0.6,
            s=marker_size,
            color=color_lookup[lab],
        )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(loc="best", markerscale=2, fontsize="small")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.close()


def run_feature_visualizations(
    train_pack: FeaturePack,
    test_pack: FeaturePack,
    embedding_key: str,
    activity_label_key: str,
    activity_label_fallbacks: Iterable[str] | None,
    method: str,
    perplexity: float,
    max_samples: int | None,
    seed: int,
    output_dir: Path,
    run_tag: str | None = None,
    id_embedding_key: str | None = None,
    activity_embedding_key: str | None = None,
    marker_size: float = 15.0,
) -> None:
    """
    Generate TSNE/PCA/UMAP projections for IDs and activities across train/test splits.
    Allows different embedding sources for ID and activity projections while keeping a
    backward-compatible default via ``embedding_key``.
    """
    base_dir = Path(output_dir)
    if run_tag:
        base_dir = base_dir / run_tag
    base_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_tag}_" if run_tag else ""

    id_key = id_embedding_key or embedding_key
    act_key = activity_embedding_key or embedding_key

    viz_specs: Sequence[tuple[str, FeaturePack, str, Iterable[str] | None, str, str]] = [
        (id_key, "train", train_pack, "global_id", None, f"{id_key}_train_id"),
        (id_key, "test", test_pack, "global_id", None, f"{id_key}_test_id"),
        (act_key, "train", train_pack, activity_label_key, activity_label_fallbacks, f"{act_key}_train_activity"),
        (act_key, "test", test_pack, activity_label_key, activity_label_fallbacks, f"{act_key}_test_activity"),
    ]
    for embed_key, split_name, pack, label_key, fallbacks, filename in viz_specs:
        subset = subsample_pack(pack, max_samples, seed)
        labels = gather_labels(subset.meta, label_key, fallbacks)
        coords = reduce_dimensionality(
            subset.embeddings[embed_key],
            method=method,
            perplexity=perplexity,
            seed=seed,
        )
        save_path = base_dir / f"{prefix}{filename}.png"
        title = f"{embed_key} {split_name} colored by {label_key}"
        plot_embedding(
            coords,
            labels,
            title=title,
            save_path=save_path,
            marker_size=float(marker_size),
        )
