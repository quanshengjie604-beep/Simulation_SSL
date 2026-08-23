from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from .feature_utils import FeaturePack, gather_labels


@dataclass
class KNNTask:
    name: str
    embedding_key: str
    label_key: str
    label_fallbacks: Iterable[str] | None = None


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return arr / denom


def _pca_whiten(train_feats: np.ndarray, test_feats: np.ndarray, method: str = "pca", eps: float = 1e-5):
    """
    PCA/ZCA whitening on train embeddings, applied to train and test.
    """

    train_mean = train_feats.mean(axis=0, keepdims=True)
    train_centered = train_feats - train_mean
    test_centered = test_feats - train_mean

    cov = np.cov(train_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    scaling = np.diag(1.0 / np.sqrt(eigvals + eps))

    if method.lower() == "zca":
        whitening = eigvecs @ scaling @ eigvecs.T
    else:  # default to PCA whitening
        whitening = eigvecs @ scaling

    train_white = train_centered @ whitening
    test_white = test_centered @ whitening
    return train_white, test_white


def _encode_labels(train_labels: List[str], test_labels: List[str]) -> tuple[np.ndarray, np.ndarray, List[str]]:
    unique = sorted(set(train_labels) | set(test_labels))
    mapping = {label: idx for idx, label in enumerate(unique)}
    train_encoded = np.array([mapping[label] for label in train_labels], dtype=np.int32)
    test_encoded = np.array([mapping[label] for label in test_labels], dtype=np.int32)
    return train_encoded, test_encoded, unique


def _knn_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Filter out skipped samples (marked with -1)
    valid_mask = y_pred != -1
    if valid_mask.sum() == 0:
        return {
            "accuracy": float("nan"),
            "balanced_acc": float("nan"),
            "macro_f1": float("nan"),
            "skipped_samples": len(y_true),
        }
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    return {
        "accuracy": accuracy_score(y_true_valid, y_pred_valid),
        "balanced_acc": balanced_accuracy_score(y_true_valid, y_pred_valid),
        "macro_f1": f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0),
        "skipped_samples": int((~valid_mask).sum()),
    }


def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, label_names: List[str]) -> Dict[str, Dict[str, float]]:
    # Filter out skipped samples (marked with -1)
    valid_mask = y_pred != -1
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    per_class: Dict[str, Dict[str, float]] = {}
    for idx, label in enumerate(label_names):
        mask = y_true_valid == idx
        total = int(mask.sum())
        if total == 0:
            per_class[label] = {"accuracy": float("nan"), "correct": 0, "total": 0}
            continue
        correct = int((y_pred_valid[mask] == idx).sum())
        per_class[label] = {
            "accuracy": correct / total if total > 0 else float("nan"),
            "correct": correct,
            "total": total,
        }
    return per_class


def _majority_vote(neighbor_labels: np.ndarray) -> int:
    """Perform majority vote on neighbor labels."""
    if len(neighbor_labels) == 0:
        raise ValueError("Cannot perform majority vote on empty neighbor list")
    counter = Counter(neighbor_labels.tolist())
    max_count = counter.most_common(1)[0][1]
    tied = sorted(label for label, cnt in counter.items() if cnt == max_count)
    return tied[0]


@dataclass
class NeighborStats:
    """Statistics about valid neighbors after track exclusion."""
    total_samples: int
    zero_valid_count: int  # samples with 0 valid neighbors (used fallback)
    reduced_count: int  # samples with >0 but <k valid neighbors
    valid_neighbor_counts: List[int]  # valid neighbor count per sample
    
    @property
    def min_valid(self) -> int:
        return min(self.valid_neighbor_counts) if self.valid_neighbor_counts else 0
    
    @property
    def max_valid(self) -> int:
        return max(self.valid_neighbor_counts) if self.valid_neighbor_counts else 0
    
    @property
    def avg_valid(self) -> float:
        return np.mean(self.valid_neighbor_counts) if self.valid_neighbor_counts else 0.0
    
    def print_summary(self, k: int, task_name: str = "activity") -> None:
        """Print detailed neighbor statistics with recommendations."""
        affected_pct = (self.zero_valid_count + self.reduced_count) / self.total_samples * 100
        print(f"\n    ⚠️  Track exclusion impact for {task_name} KNN:")
        print(f"       Samples with 0 valid neighbors (used fallback): {self.zero_valid_count}/{self.total_samples}")
        print(f"       Samples with <k valid neighbors: {self.reduced_count}/{self.total_samples}")
        print(f"       Total affected: {self.zero_valid_count + self.reduced_count}/{self.total_samples} ({affected_pct:.1f}%)")
        print(f"       Valid neighbors - Min: {self.min_valid}, Max: {self.max_valid}, Avg: {self.avg_valid:.1f}")
        
        if self.zero_valid_count > 0:
            # Recommend a k value that would leave most samples with valid neighbors
            recommended_k = max(k * 2, int(np.ceil(k / (1 - self.zero_valid_count / self.total_samples))) + 5)
            print(f"       💡 Recommendation: Increase k from {k} to at least {recommended_k} for reliable exclude_same_track results")


def _extract_velocity_vector(meta: dict, speed_feature_names: List[str]) -> np.ndarray | None:
    """Extract velocity vector from metadata using specified feature names or fallbacks."""
    # Try primary feature names first
    for feat_name in speed_feature_names:
        if feat_name in meta:
            vx = meta[feat_name]
            if not (isinstance(vx, (int, float)) and not np.isnan(vx)):
                continue
            # Look for corresponding y component
            vy_name = feat_name.replace("_x", "_y").replace("x", "y")
            if vy_name in meta:
                vy = meta[vy_name]
                if isinstance(vy, (int, float)) and not np.isnan(vy):
                    return np.array([float(vx), float(vy)], dtype=np.float32)
    
    # Fallback to mean velocity fields
    if "vx_mean" in meta and "vy_mean" in meta:
        vx = meta["vx_mean"]
        vy = meta["vy_mean"]
        if isinstance(vx, (int, float)) and isinstance(vy, (int, float)):
            if not (np.isnan(vx) or np.isnan(vy)):
                return np.array([float(vx), float(vy)], dtype=np.float32)
    
    return None


def _run_knn_with_exclusion(
    knn: KNeighborsClassifier,
    train_y: np.ndarray,
    test_feats: np.ndarray,
    test_y: np.ndarray,
    k: int,
    train_track_ids: List[str] | None = None,
    test_track_ids: List[str] | None = None,
    exclude_same_track: bool = False,
    train_meta: List[dict] | None = None,
    test_meta: List[dict] | None = None,
    exclude_similar_motion: bool = False,
    speed_tolerance: float = 1.5,
    direction_min_cosine: float = 0.866,
    speed_feature_names: List[str] | None = None,
    zero_neighbors_fallback: str = "use_all",
) -> Tuple[np.ndarray, NeighborStats | None]:
    """
    Run KNN prediction with optional same-track or similar-motion exclusion.
    Returns predictions and neighbor statistics (if exclusion is applied).
    """
    if not (exclude_same_track or exclude_similar_motion):
        return knn.predict(test_feats), None
    
    # Get neighbor indices for all test samples
    neighbor_distances, neighbor_indices = knn.kneighbors(test_feats, n_neighbors=k)
    
    # Extract velocity vectors if motion filtering is enabled
    train_velocities = None
    test_velocities = None
    if exclude_similar_motion:
        if train_meta is None or test_meta is None:
            raise ValueError("train_meta and test_meta required for motion-based exclusion")
        speed_feature_names = speed_feature_names or ["v_x", "v_y"]
        train_velocities = [_extract_velocity_vector(meta, speed_feature_names) for meta in train_meta]
        test_velocities = [_extract_velocity_vector(meta, speed_feature_names) for meta in test_meta]
        # Check if we have enough velocity data
        train_has_vel = sum(1 for v in train_velocities if v is not None)
        test_has_vel = sum(1 for v in test_velocities if v is not None)
        if train_has_vel == 0 or test_has_vel == 0:
            print(f"    Warning: Motion filtering requested but insufficient velocity data "
                  f"(train: {train_has_vel}/{len(train_velocities)}, test: {test_has_vel}/{len(test_velocities)}). "
                  f"Falling back to standard KNN.")
            return knn.predict(test_feats), None
    
    # Track statistics
    zero_valid_count = 0
    reduced_count = 0
    valid_neighbor_counts = []
    
    # Filter neighbors and perform predictions
    preds = np.zeros(len(test_feats), dtype=np.int32)
    for i in range(len(test_feats)):
        neighbors = neighbor_indices[i]
        valid_neighbors = []
        
        for j in neighbors:
            # Track-based exclusion
            if exclude_same_track and train_track_ids is not None and test_track_ids is not None:
                if train_track_ids[j] == test_track_ids[i]:
                    continue  # Skip same track
            
            # Motion-based exclusion
            if exclude_similar_motion and train_velocities[j] is not None and test_velocities[i] is not None:
                train_vel = train_velocities[j]
                test_vel = test_velocities[i]
                
                # Compute speed difference (L2 norm)
                vel_diff = np.linalg.norm(train_vel - test_vel)
                if vel_diff <= speed_tolerance:
                    # Check direction similarity (cosine)
                    if direction_min_cosine > -1.0:
                        dot_product = np.dot(train_vel, test_vel)
                        train_norm = np.linalg.norm(train_vel)
                        test_norm = np.linalg.norm(test_vel)
                        if train_norm > 1e-9 and test_norm > 1e-9:
                            cosine = dot_product / (train_norm * test_norm)
                            if cosine >= direction_min_cosine:
                                continue  # Skip similar motion (both speed and direction match)
                    else:
                        continue  # Skip if speed matches (direction check disabled)
            
            valid_neighbors.append(j)
        
        num_valid = len(valid_neighbors)
        valid_neighbor_counts.append(num_valid)
        
        if num_valid == 0:
            zero_valid_count += 1
            if zero_neighbors_fallback == "skip":
                # Mark as invalid (will be excluded from metrics)
                preds[i] = -1  # Use -1 as sentinel value for skipped samples
                continue
            elif zero_neighbors_fallback == "expand_k":
                # Try to find more neighbors by expanding search
                expanded_k = min(k * 3, len(train_y))  # Try up to 3x k, but not more than train size
                if expanded_k > k:
                    expanded_distances, expanded_indices = knn.kneighbors(
                        test_feats[i:i+1], n_neighbors=expanded_k
                    )
                    expanded_neighbors = expanded_indices[0]
                    # Re-filter with expanded neighbors
                    valid_neighbors = []
                    for j in expanded_neighbors:
                        skip = False
                        if exclude_same_track and train_track_ids is not None and test_track_ids is not None:
                            if train_track_ids[j] == test_track_ids[i]:
                                skip = True
                        if not skip and exclude_similar_motion and train_velocities[j] is not None and test_velocities[i] is not None:
                            train_vel = train_velocities[j]
                            test_vel = test_velocities[i]
                            vel_diff = np.linalg.norm(train_vel - test_vel)
                            if vel_diff <= speed_tolerance:
                                if direction_min_cosine > -1.0:
                                    dot_product = np.dot(train_vel, test_vel)
                                    train_norm = np.linalg.norm(train_vel)
                                    test_norm = np.linalg.norm(test_vel)
                                    if train_norm > 1e-9 and test_norm > 1e-9:
                                        cosine = dot_product / (train_norm * test_norm)
                                        if cosine >= direction_min_cosine:
                                            skip = True
                                else:
                                    skip = True
                        if not skip:
                            valid_neighbors.append(j)
                    
                    if len(valid_neighbors) == 0:
                        # Still no valid neighbors after expansion, fall back to use_all
                        valid_neighbors = neighbors.tolist()
                    else:
                        num_valid = len(valid_neighbors)
                        valid_neighbor_counts[-1] = num_valid  # Update count
            else:  # "use_all" (default)
                # Fallback: use all neighbors if filtering removes everything
                valid_neighbors = neighbors.tolist()
        elif num_valid < k:
            reduced_count += 1
        
        # Majority vote on valid neighbors
        neighbor_labels = train_y[valid_neighbors]
        preds[i] = _majority_vote(neighbor_labels)
    
    stats = NeighborStats(
        total_samples=len(test_feats),
        zero_valid_count=zero_valid_count,
        reduced_count=reduced_count,
        valid_neighbor_counts=valid_neighbor_counts,
    )
    
    return preds, stats


def run_knn_dual_exclusion(
    train_pack: FeaturePack,
    test_pack: FeaturePack,
    task: KNNTask,
    k: int,
    normalize: bool,
    whiten: dict | None = None,
    track_key: str = "global_id",
    exclude_similar_motion: bool = False,
    speed_tolerance: float = 1.5,
    direction_min_cosine: float = 0.866,
    speed_feature_names: List[str] | None = None,
    zero_neighbors_fallback: str = "use_all",
) -> Dict[str, Dict[str, float]]:
    """
    Run KNN for a single task with BOTH exclude_same_track=True and False.
    Prints side-by-side comparison and detailed neighbor statistics.
    
    Returns dict with keys 'include_same_track' and 'exclude_same_track'.
    """
    results: Dict[str, Dict[str, float]] = {}
    
    whiten_enabled = False
    whiten_method = "pca"
    whiten_eps = 1e-5
    if whiten is not None:
        whiten_enabled = bool(getattr(whiten, "enabled", False))
        whiten_method = str(getattr(whiten, "method", "pca")).lower()
        whiten_eps = float(getattr(whiten, "eps", 1e-5))
    
    # Prepare features and labels
    train_labels = gather_labels(train_pack.meta, task.label_key, task.label_fallbacks)
    test_labels = gather_labels(test_pack.meta, task.label_key, task.label_fallbacks)
    train_y, test_y, label_names = _encode_labels(train_labels, test_labels)
    train_feats = train_pack.embeddings[task.embedding_key]
    test_feats = test_pack.embeddings[task.embedding_key]
    
    if normalize:
        train_feats = _l2_normalize(train_feats)
        test_feats = _l2_normalize(test_feats)
    
    if whiten_enabled:
        train_feats, test_feats = _pca_whiten(train_feats, test_feats, method=whiten_method, eps=whiten_eps)
        train_feats = _l2_normalize(train_feats)
        test_feats = _l2_normalize(test_feats)
    
    # Get track IDs for exclusion
    train_track_ids = gather_labels(train_pack.meta, track_key, fallbacks=None)
    test_track_ids = gather_labels(test_pack.meta, track_key, fallbacks=None)
    
    # Fit KNN
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", algorithm="brute")
    knn.fit(train_feats, train_y)
    
    # Run without exclusion (include same track)
    preds_include, _ = _run_knn_with_exclusion(
        knn, train_y, test_feats, test_y, k,
        train_track_ids=None, test_track_ids=None,
        exclude_same_track=False,
        zero_neighbors_fallback="use_all",
    )
    metrics_include = _knn_metrics(test_y, preds_include)
    metrics_include["per_class"] = _per_class_accuracy(test_y, preds_include, label_names)
    results["include_same_track"] = metrics_include
    
    # Run with exclusion (exclude same track and/or similar motion)
    preds_exclude, neighbor_stats = _run_knn_with_exclusion(
        knn, train_y, test_feats, test_y, k,
        train_track_ids=train_track_ids, test_track_ids=test_track_ids,
        exclude_same_track=True,
        train_meta=train_pack.meta,
        test_meta=test_pack.meta,
        exclude_similar_motion=exclude_similar_motion,
        speed_tolerance=speed_tolerance,
        direction_min_cosine=direction_min_cosine,
        speed_feature_names=speed_feature_names,
        zero_neighbors_fallback=zero_neighbors_fallback,
    )
    metrics_exclude = _knn_metrics(test_y, preds_exclude)
    metrics_exclude["per_class"] = _per_class_accuracy(test_y, preds_exclude, label_names)
    results["exclude_same_track"] = metrics_exclude

    return results


def run_knn_probes(
    train_pack: FeaturePack,
    test_pack: FeaturePack,
    tasks: Iterable[KNNTask],
    k: int,
    normalize: bool,
    whiten: dict | None = None,
    exclude_same_track: bool = False,
    track_key: str = "global_id",
    exclude_similar_motion: bool = False,
    speed_tolerance: float = 1.5,
    direction_min_cosine: float = 0.866,
    speed_feature_names: List[str] | None = None,
    zero_neighbors_fallback: str = "use_all",
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    whiten_enabled = False
    whiten_method = "pca"
    whiten_eps = 1e-5
    if whiten is not None:
        whiten_enabled = bool(getattr(whiten, "enabled", False))
        whiten_method = str(getattr(whiten, "method", "pca")).lower()
        whiten_eps = float(getattr(whiten, "eps", 1e-5))

    for task in tasks:
        train_labels = gather_labels(train_pack.meta, task.label_key, task.label_fallbacks)
        test_labels = gather_labels(test_pack.meta, task.label_key, task.label_fallbacks)
        train_y, test_y, label_names = _encode_labels(train_labels, test_labels)
        train_feats = train_pack.embeddings[task.embedding_key]
        test_feats = test_pack.embeddings[task.embedding_key]

        if normalize:
            train_feats = _l2_normalize(train_feats)
            test_feats = _l2_normalize(test_feats)

        if whiten_enabled:
            train_feats, test_feats = _pca_whiten(train_feats, test_feats, method=whiten_method, eps=whiten_eps)
            # Re-normalize after whitening
            train_feats = _l2_normalize(train_feats)
            test_feats = _l2_normalize(test_feats)

        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean", algorithm="brute")
        knn.fit(train_feats, train_y)

        # Apply exclusion only for activity tasks
        apply_track_exclusion = exclude_same_track and task.name == "activity"
        apply_motion_exclusion = exclude_similar_motion and task.name == "activity"
        
        if apply_track_exclusion or apply_motion_exclusion:
            # Extract track IDs if needed
            train_track_ids = None
            test_track_ids = None
            if apply_track_exclusion:
                train_track_ids = gather_labels(train_pack.meta, track_key, fallbacks=None)
                test_track_ids = gather_labels(test_pack.meta, track_key, fallbacks=None)
            
            preds, neighbor_stats = _run_knn_with_exclusion(
                knn, train_y, test_feats, test_y, k,
                train_track_ids=train_track_ids,
                test_track_ids=test_track_ids,
                exclude_same_track=apply_track_exclusion,
                train_meta=train_pack.meta,
                test_meta=test_pack.meta,
                exclude_similar_motion=apply_motion_exclusion,
                speed_tolerance=speed_tolerance,
                direction_min_cosine=direction_min_cosine,
                speed_feature_names=speed_feature_names,
                zero_neighbors_fallback=zero_neighbors_fallback,
            )
            
            if neighbor_stats is not None:
                neighbor_stats.print_summary(k, task.name)
        else:
            preds = knn.predict(test_feats)
            neighbor_stats = None

        metrics = _knn_metrics(test_y, preds)
        metrics["per_class"] = _per_class_accuracy(test_y, preds, label_names)
        results[task.name] = metrics
        
        exclusion_msgs = []
        if apply_track_exclusion:
            exclusion_msgs.append("excl. same-track")
        if apply_motion_exclusion:
            exclusion_msgs.append("excl. similar-motion")
        exclusion_msg = f" ({', '.join(exclusion_msgs)})" if exclusion_msgs else ""
        
        print(
            f"[KNN:{task.name}] k={k}{exclusion_msg} | Acc={metrics['accuracy']:.4f} "
            f"| BalAcc={metrics['balanced_acc']:.4f} | Macro-F1={metrics['macro_f1']:.4f}"
        )
        # Identity retrieval (global_id) produces a very long list of IDs, so skip detailed per-class logging.
        if str(task.label_key).lower() != "global_id":
            class_lines = []
            for label, stats in metrics["per_class"].items():
                total = stats["total"]
                if total == 0:
                    class_lines.append(f"{label}: N/A")
                else:
                    class_lines.append(f"{label}: {stats['accuracy']:.3f} ({stats['correct']}/{total})")
            if class_lines:
                print("    Per-class accuracy -> " + " | ".join(class_lines))
    return results
