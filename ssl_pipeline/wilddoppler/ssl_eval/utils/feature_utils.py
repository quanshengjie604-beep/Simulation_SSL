from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class FeaturePack:
    """
    Container holding the embeddings and metadata for a dataset split.

    embeddings:
        Mapping from embedding name (e.g., "embedding", "z_id") to a [N, D] numpy array.
    meta:
        List of metadata dicts per sample. Each entry should at least include the
        activity and global_id fields required for downstream evaluations.
    """

    embeddings: Dict[str, np.ndarray]
    meta: List[dict]

    def __post_init__(self) -> None:
        if not self.embeddings:
            raise ValueError("FeaturePack requires at least one embedding entry.")
        sample_counts = {name: arr.shape[0] for name, arr in self.embeddings.items()}
        first = next(iter(sample_counts.values()))
        if any(count != first for count in sample_counts.values()):
            raise ValueError(f"Inconsistent sample counts in FeaturePack: {sample_counts}")

    @property
    def num_samples(self) -> int:
        return next(iter(self.embeddings.values())).shape[0]

    def subset(self, indices: Sequence[int]) -> "FeaturePack":
        idx = np.array(indices, dtype=int)
        new_embeddings = {name: emb[idx] for name, emb in self.embeddings.items()}
        new_meta = [self.meta[i] for i in idx]
        return FeaturePack(embeddings=new_embeddings, meta=new_meta)


def unpack_meta(meta_obj) -> List[dict]:
    """
    Convert the dataset metadata (list of dicts or dict of tensors) into a list of dicts.
    """

    if isinstance(meta_obj, list):
        return [dict(entry) for entry in meta_obj]
    if not isinstance(meta_obj, Mapping):
        raise TypeError(f"Unsupported meta container type: {type(meta_obj)}")
    entries: List[dict] = []
    batch_size = None
    for key, value in meta_obj.items():
        values = _value_to_list(value)
        if batch_size is None:
            batch_size = len(values)
            entries = [{} for _ in range(batch_size)]
        elif len(values) != batch_size:
            raise ValueError(f"Inconsistent batch dimension for meta key '{key}'.")
        for idx, entry in enumerate(entries):
            entry[key] = values[idx]
    return entries


def _value_to_list(value) -> List:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu()
        if arr.ndim == 0:
            return [arr.item()]
        return arr.tolist()
    if isinstance(value, np.ndarray):
        arr = value
        if value.ndim == 0:
            return [arr.item()]
        return arr.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_eval_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    embedding_keys: Sequence[str],
    split_name: str,
) -> FeaturePack:
    banks: MutableMapping[str, List[np.ndarray]] = {key: [] for key in embedding_keys}
    meta_entries: List[dict] = []
    with torch.no_grad():
        for x, _, meta in loader:
            x = x.to(device, dtype=torch.float32)
            outputs = model(x)
            for key in embedding_keys:
                if key not in outputs:
                    raise KeyError(f"Embedding key '{key}' missing from model outputs.")
                banks[key].append(outputs[key].detach().cpu().numpy())
            entries = unpack_meta(meta)
            for entry in entries:
                entry.setdefault("split", split_name)
            meta_entries.extend(entries)
    embeddings = {key: np.concatenate(parts, axis=0) for key, parts in banks.items()}
    return FeaturePack(embeddings=embeddings, meta=meta_entries)


def concat_feature_packs(packs: Sequence[FeaturePack]) -> FeaturePack:
    if not packs:
        raise ValueError("concat_feature_packs requires at least one pack.")
    embedding_names = set(packs[0].embeddings.keys())
    for pack in packs[1:]:
        if set(pack.embeddings.keys()) != embedding_names:
            raise ValueError("All FeaturePacks must share the same embedding keys.")
    concatenated = {
        name: np.concatenate([pack.embeddings[name] for pack in packs], axis=0)
        for name in embedding_names
    }
    meta: List[dict] = []
    for pack in packs:
        meta.extend(pack.meta)
    return FeaturePack(embeddings=concatenated, meta=meta)


def gather_labels(meta_list: Sequence[dict], key: str, fallbacks: Iterable[str] | None = None) -> List[str]:
    labels: List[str] = []
    for meta in meta_list:
        value = meta.get(key)
        if value is None and fallbacks:
            for alt in fallbacks:
                if alt in meta:
                    value = meta[alt]
                    break
        labels.append(str(value))
    return labels


def center_pack_by_id(
    pack: FeaturePack,
    embedding_key: str,
    id_key: str,
    id_fallbacks: Iterable[str] | None = None,
) -> FeaturePack:
    """
    Subtract the per-ID mean embedding from each sample in the FeaturePack.
    Does not modify the input pack; returns a new pack with centered embeddings.
    """
    ids = gather_labels(pack.meta, id_key, id_fallbacks)
    feats = pack.embeddings[embedding_key]
    if feats.shape[0] != len(ids):
        raise ValueError(f"ID count ({len(ids)}) does not match embeddings ({feats.shape[0]}).")

    centered = feats.copy()
    ids_arr = np.asarray(ids)
    for gid in np.unique(ids_arr):
        mask = ids_arr == gid
        centered[mask] = feats[mask] - feats[mask].mean(axis=0, keepdims=True)

    new_embeddings = dict(pack.embeddings)
    new_embeddings[embedding_key] = centered
    return FeaturePack(embeddings=new_embeddings, meta=pack.meta)


def subsample_pack(pack: FeaturePack, max_samples: int | None, seed: int) -> FeaturePack:
    if max_samples is None or pack.num_samples <= max_samples:
        return pack
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(pack.num_samples, size=max_samples, replace=False).tolist())
    return pack.subset(idx)
