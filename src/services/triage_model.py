"""
XLM-R tabanli cok-basli triage siniflandirici.

Gorevler:
  1. Triage level    (CRITICAL / URGENT / NON_URGENT)           - 3-sinif CE
  2. Category        (medical / fire / crime / other)           - 4-sinif CE
  3. Red flag        (yok / var)                                - binary BCE
  4. Input quality   (meaningful / gibberish / out_of_scope)    - 3-sinif CE

Hem `scripts/train_triage_xlmr.py` hem de `src/services/triage_local_service.py`
bu modul uzerinden modeli yukler.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


TRIAGE_LABELS: List[str] = ["CRITICAL", "URGENT", "NON_URGENT"]
CATEGORY_LABELS: List[str] = ["medical", "fire", "crime", "other"]
QUALITY_LABELS: List[str] = ["meaningful", "gibberish", "out_of_scope"]
DEFAULT_BASE_MODEL = "xlm-roberta-base"
DEFAULT_MAX_LEN = 384


class XlmrMultiTaskTriage(nn.Module):
    """XLM-RoBERTa encoder + 4 classification heads."""

    def __init__(
        self,
        base_model_name: str = DEFAULT_BASE_MODEL,
        n_triage: int = 3,
        n_category: int = 4,
        n_quality: int = 3,
        dropout: float = 0.1,
        init_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.base_model_name = base_model_name
        self.n_triage = n_triage
        self.n_category = n_category
        self.n_quality = n_quality

        if init_encoder:
            from transformers import AutoModel  # type: ignore
            self.encoder = AutoModel.from_pretrained(base_model_name)
        else:
            from transformers import AutoConfig, AutoModel  # type: ignore
            cfg = AutoConfig.from_pretrained(base_model_name)
            self.encoder = AutoModel.from_config(cfg)

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_triage = nn.Linear(hidden, n_triage)
        self.head_category = nn.Linear(hidden, n_category)
        self.head_redflag = nn.Linear(hidden, 1)
        self.head_quality = nn.Linear(hidden, n_quality)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        pooled = self.dropout(pooled)
        return {
            "logits_triage": self.head_triage(pooled),
            "logits_category": self.head_category(pooled),
            "logits_redflag": self.head_redflag(pooled).squeeze(-1),
            "logits_quality": self.head_quality(pooled),
        }

    def save(self, save_dir: Path | str, tokenizer: Any | None = None,
             extra_meta: Dict[str, Any] | None = None) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_dir / "model.pt")
        meta: Dict[str, Any] = {
            "base_model_name": self.base_model_name,
            "n_triage": self.n_triage,
            "n_category": self.n_category,
            "n_quality": self.n_quality,
            "triage_labels": TRIAGE_LABELS,
            "category_labels": CATEGORY_LABELS,
            "quality_labels": QUALITY_LABELS,
            "model_type": "xlmr_multitask_triage",
            "model_version": 2,
            "max_len": DEFAULT_MAX_LEN,
        }
        if extra_meta:
            meta.update(extra_meta)
        with (save_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        if tokenizer is not None:
            tok_dir = save_dir / "tokenizer"
            tok_dir.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(tok_dir))

    @classmethod
    def load(cls, save_dir: Path | str, device: torch.device | None = None) -> "XlmrMultiTaskTriage":
        save_dir = Path(save_dir)
        with (save_dir / "config.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        model = cls(
            base_model_name=meta.get("base_model_name", DEFAULT_BASE_MODEL),
            n_triage=int(meta.get("n_triage", 3)),
            n_category=int(meta.get("n_category", 4)),
            n_quality=int(meta.get("n_quality", 3)),
            init_encoder=False,
        )
        state = torch.load(save_dir / "model.pt", map_location=device or "cpu")
        # Backward-compat: eski 3-basli checkpoint yuklenirse head_quality henuz
        # rastgele; strict=False ile izin ver ve uyari logla.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if any("head_quality" in k for k in missing):
            logger.warning(
                "Legacy checkpoint: head_quality agirliklari eksik, rastgele "
                "baslatildi (versiyon 1 checkpoint)."
            )
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)
        model.eval()
        if device is not None:
            model.to(device)
        return model


def load_tokenizer(save_dir: Path | str):
    from transformers import AutoTokenizer  # type: ignore
    tok_dir = Path(save_dir) / "tokenizer"
    if tok_dir.exists():
        return AutoTokenizer.from_pretrained(str(tok_dir))
    logger.warning("Tokenizer dir not found at %s; falling back to base.", tok_dir)
    return AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
