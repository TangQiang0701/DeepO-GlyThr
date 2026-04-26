from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .constants import AA_PHYSICOCHEM, AA_TO_IDX, AMINO_ACIDS, CHANNEL_NAMES, PHYS_NAMES
from .model import CNNBiGRUAttentionNet


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "model.pt"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_MODEL = None


@dataclass
class WindowRecord:
    source_id: str
    window_id: str
    source_length: int
    site_position: int
    mode: str
    sequence: str


def zscore_physchem_table(table: Dict[str, Sequence[float]]) -> Dict[str, np.ndarray]:
    aas = sorted(table)
    mat = np.array([table[a] for a in aas], dtype=np.float32)
    mat = (mat - mat.mean(0, keepdims=True)) / (mat.std(0, keepdims=True) + 1e-8)
    return {a: v for a, v in zip(aas, mat)}


AA_PHYS_Z = zscore_physchem_table(AA_PHYSICOCHEM)


def load_model(model_path: Path | None = None):
    global _MODEL
    if _MODEL is None:
        checkpoint = torch.load(model_path or MODEL_PATH, map_location=DEVICE)
        model = CNNBiGRUAttentionNet()
        model.load_state_dict(checkpoint["model_state_dict"])
        _MODEL = model.to(DEVICE)
        _MODEL.eval()
    return _MODEL


def encode_seq(seq: str) -> np.ndarray:
    seq = seq.upper()
    length = len(seq)
    center = length // 2
    onehot = np.zeros((length, 20), dtype=np.float32)
    phys = np.zeros((length, 6), dtype=np.float32)
    pos_prior = np.zeros((length, 1), dtype=np.float32)
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            onehot[i, AA_TO_IDX[aa]] = 1.0
        if aa in AA_PHYS_Z:
            phys[i] = AA_PHYS_Z[aa]
        pos_prior[i, 0] = 1.0 - abs(i - center) / max(center, 1)
    return np.concatenate([onehot, phys, pos_prior], axis=1)


def normalize_text_input(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("Input is empty.")
    if text.startswith(">"):
        return text
    seq = "".join(text.split()).upper()
    return f">query\n{seq}"


def parse_fasta_text(text: str) -> List[Tuple[str, str]]:
    text = normalize_text_input(text)
    records: List[Tuple[str, str]] = []
    header = None
    seq_parts: List[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts).upper()))
            header = line[1:].strip() or f"sequence_{len(records) + 1}"
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts).upper()))
    if not records:
        raise ValueError("No FASTA records were found.")
    return records


def is_valid_sequence(seq: str) -> bool:
    return all(char in AMINO_ACIDS for char in seq.upper())


def extract_windows(source_id: str, sequence: str) -> List[WindowRecord]:
    sequence = "".join(sequence.split()).upper()
    if not is_valid_sequence(sequence):
        raise ValueError(f"Record {source_id} contains unsupported amino acids.")
    if len(sequence) < 41:
        raise ValueError(f"Record {source_id} is shorter than 41 residues.")

    windows: List[WindowRecord] = []
    if len(sequence) == 41:
        if sequence[20] != "T":
            raise ValueError(f"Record {source_id} must contain T at position 21 for 41-aa mode.")
        windows.append(
            WindowRecord(
                source_id=source_id,
                window_id=source_id,
                source_length=41,
                site_position=21,
                mode="window",
                sequence=sequence,
            )
        )
        return windows

    for idx, aa in enumerate(sequence):
        if aa != "T":
            continue
        if idx < 20 or idx > len(sequence) - 21:
            continue
        windows.append(
            WindowRecord(
                source_id=source_id,
                window_id=f"{source_id}_{idx + 1}",
                source_length=len(sequence),
                site_position=idx + 1,
                mode="scan",
                sequence=sequence[idx - 20 : idx + 21],
            )
        )
    if not windows:
        raise ValueError(
            f"Record {source_id} does not contain a valid threonine with at least 20 upstream and 20 downstream residues."
        )
    return windows


def parse_input_text(text: str) -> List[WindowRecord]:
    records = parse_fasta_text(text)
    windows: List[WindowRecord] = []
    errors = []
    for source_id, sequence in records:
        try:
            windows.extend(extract_windows(source_id, sequence))
        except ValueError as exc:
            errors.append(str(exc))
    if not windows:
        raise ValueError("\n".join(errors) if errors else "No valid windows were found.")
    return windows


@torch.no_grad()
def predict_with_attention(sequences: Sequence[str], batch_size: int = 256) -> Tuple[List[float], List[List[float]]]:
    model = load_model()
    feats = np.stack([encode_seq(seq) for seq in sequences], axis=0)
    x = torch.tensor(feats, dtype=torch.float32)
    probs_all: List[float] = []
    attn_all: List[List[float]] = []
    for i in range(0, len(x), batch_size):
        xb = x[i : i + batch_size].to(DEVICE)
        logits, attn_weights = model(xb)
        probs_all.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        attn_all.extend(attn_weights.cpu().numpy().tolist())
    return probs_all, attn_all


def predict_windows(windows: Sequence[WindowRecord]) -> List[Dict]:
    probs, attentions = predict_with_attention([window.sequence for window in windows])
    rows = []
    for window, prob, attention in zip(windows, probs, attentions):
        center = len(window.sequence) // 2
        top_idx = sorted(range(len(attention)), key=lambda idx: attention[idx], reverse=True)[:3]
        rows.append(
            {
                "source_id": window.source_id,
                "window_id": window.window_id,
                "source_length": window.source_length,
                "site_position": window.site_position,
                "mode": window.mode,
                "sequence": window.sequence,
                "probability": round(float(prob), 6),
                "prediction": "Positive" if float(prob) >= 0.5 else "Negative",
                "attention": [round(float(item), 6) for item in attention],
                "top_attention_positions": [idx - center for idx in top_idx],
            }
        )
    return rows


def build_summary_dataframe(rows: Sequence[Dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "window_id": row["window_id"],
                "source_id": row["source_id"],
                "site_position": row["site_position"],
                "mode": row["mode"],
                "probability": row["probability"],
                "prediction": row["prediction"],
                "top_attention_positions": ", ".join(str(x) for x in row["top_attention_positions"]),
                "sequence": row["sequence"],
            }
            for row in rows
        ]
    )


def _predict_prob_from_tensor(model, x: torch.Tensor) -> torch.Tensor:
    logits, _ = model(x)
    return torch.sigmoid(logits)


@torch.no_grad()
def _predict_prob_from_features(model, feats: np.ndarray) -> float:
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return float(_predict_prob_from_tensor(model, x).item())


def _mask_features(feats: np.ndarray, group_name: str) -> np.ndarray:
    masked = feats.copy()
    if group_name == "one_hot":
        masked[:, :20] = 0.0
    elif group_name == "physicochemical":
        masked[:, 20:26] = 0.0
    elif group_name == "center_prior":
        masked[:, 26] = 0.0
    elif group_name in PHYS_NAMES:
        masked[:, 20 + PHYS_NAMES.index(group_name)] = 0.0
    return masked


def integrated_gradients(seq: str, steps: int = 32) -> Tuple[float, np.ndarray]:
    model = load_model()
    x = torch.tensor(encode_seq(seq), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)

    for alpha in torch.linspace(0, 1, steps + 1, device=DEVICE)[1:]:
        scaled = (baseline + alpha * (x - baseline)).detach().clone()
        scaled.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        prob = _predict_prob_from_tensor(model, scaled)
        prob.backward(torch.ones_like(prob))
        total_grad += scaled.grad

    avg_grad = total_grad / steps
    ig = ((x - baseline) * avg_grad).detach().cpu().numpy()[0]
    probability = float(_predict_prob_from_tensor(model, x).item())
    return probability, ig


def summarize_ig(seq: str, ig_attr: np.ndarray) -> Dict:
    position_scores = np.sum(np.abs(ig_attr), axis=1)
    feature_scores = [{"name": "Amino-acid identity", "importance": float(np.sum(np.abs(ig_attr[:, :20])))}]
    for idx, phys_name in enumerate(PHYS_NAMES):
        feature_scores.append(
            {
                "name": phys_name.replace("_", " ").title(),
                "importance": float(np.sum(np.abs(ig_attr[:, 20 + idx]))),
            }
        )
    feature_scores.append({"name": "Center prior", "importance": float(np.sum(np.abs(ig_attr[:, 26])))})
    feature_scores.sort(key=lambda item: item["importance"], reverse=True)

    rel_positions = [idx - (len(seq) // 2) for idx in range(len(seq))]
    top_positions = sorted(
        zip(rel_positions, list(seq), position_scores.tolist()),
        key=lambda item: item[2],
        reverse=True,
    )[:6]
    return {
        "relative_positions": rel_positions,
        "position_scores": [round(float(value), 6) for value in position_scores.tolist()],
        "feature_scores": [
            {"name": item["name"], "importance": round(float(item["importance"]), 6)}
            for item in feature_scores
        ],
        "top_positions": [(int(pos), aa, round(float(score), 6)) for pos, aa, score in top_positions],
    }


def single_sequence_ablation(seq: str) -> List[Dict]:
    model = load_model()
    feats = encode_seq(seq)
    full_probability = _predict_prob_from_features(model, feats)
    group_order = [
        ("Full model", None),
        ("w/o center prior", "center_prior"),
        ("w/o one-hot", "one_hot"),
        ("w/o physicochemical", "physicochemical"),
        ("w/o hydrophobicity", "hydrophobicity"),
        ("w/o polarity", "polarity"),
        ("w/o charge", "charge"),
        ("w/o volume", "volume"),
        ("w/o aromaticity", "aromaticity"),
        ("w/o flexibility", "flexibility"),
    ]

    rows = []
    for label, key in group_order:
        probability = full_probability if key is None else _predict_prob_from_features(model, _mask_features(feats, key))
        rows.append(
            {
                "name": label,
                "probability": round(probability, 6),
                "delta": round(full_probability - probability, 6),
            }
        )
    rows.sort(key=lambda item: item["delta"], reverse=True)
    full_row = next(item for item in rows if item["name"] == "Full model")
    rows.remove(full_row)
    rows.insert(0, full_row)
    return rows


def explain_window(row: Dict) -> Dict:
    probability, ig_attr = integrated_gradients(row["sequence"])
    return {
        "probability": round(probability, 6),
        "attention": row["attention"],
        "ig": summarize_ig(row["sequence"], ig_attr),
        "ablation": single_sequence_ablation(row["sequence"]),
    }


def rows_to_json(rows: Sequence[Dict]) -> str:
    return json.dumps(rows, indent=2)
