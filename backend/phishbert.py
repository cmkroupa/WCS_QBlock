"""
phishbert.py – PhishBERT classifier and transformer utilities.

Exports:
    PhishBERTClassifier  — XLM-RoBERTa + linear head, fine-tuned for phishing.
                           sklearn-compatible: fit / predict_proba / save / load.
    combine_texts        — concatenate visible text + structural core for input.
    get_device           — select CUDA / MPS / CPU automatically.
    get_n_gpu            — number of usable CUDA GPUs.
    get_transformer      — load a raw HuggingFace transformer (used by train.py).
    embed_texts          — mean-pool embeddings from a loaded transformer.
"""

import numpy as np
from tqdm.auto import tqdm

TRANSFORMER_NAME = "xlm-roberta-base"
EMB_BATCH_SIZE   = 32   # 32 fills MPS/CUDA pipelines better than 16; safe on CPU too


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_n_gpu():
    """Return number of usable CUDA GPUs (0 if none)."""
    import torch
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


# ─────────────────────────────────────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────────────────────────────────────

def combine_texts(vis_text: str, struct_core: str) -> str:
    """Concatenate visible text + structural fingerprint for classifier input."""
    parts = [p.strip() for p in (vis_text, struct_core) if p and p.strip()]
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# PhishBERT — fine-tuned RoBERTa classification head (Voter B)
# ─────────────────────────────────────────────────────────────────────────────

class _PhishBERTDataset:
    """Minimal PyTorch-compatible dataset — stores raw texts, tokenises per batch."""

    def __init__(self, texts, labels=None):
        self.texts  = list(texts)
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["labels"] = float(self.labels[idx])
        return item


class PhishBERTClassifier:
    """
    XLM-RoBERTa + linear classification head, fine-tuned for phishing detection.

    Architecture
    ────────────
      backbone  : xlm-roberta-base (bottom layers frozen)
      pooling   : [CLS] token from last hidden state (position 0)
      head      : Dropout(p) → Linear(hidden_size, 1)
      loss      : BCEWithLogitsLoss with pos_weight for class imbalance

    Training modes (n_unfreeze_layers)
    ────────────────────────────────────
      0  — head only  (frozen backbone, ~seconds per epoch — used for OOF proxy)
      2  — fine-tune top-2 transformer blocks + head  (default)

    Sklearn-compatible: fit(texts, y), predict_proba(texts) → (N, 2).
    Input texts should be combine_texts(vis_text, struct_core).
    """

    def __init__(
        self,
        model_name=TRANSFORMER_NAME,
        n_unfreeze_layers=2,
        dropout=0.2,
        lr=2e-5,
        head_lr=5e-4,
        weight_decay=0.01,
        batch_size=32,
        max_epochs=3,
        patience=2,
        warmup_ratio=0.06,
        max_length=256,
        device=None,
        random_state=42,
    ):
        self.model_name        = model_name
        self.n_unfreeze_layers = n_unfreeze_layers
        self.dropout           = dropout
        self.lr                = lr
        self.head_lr           = head_lr
        self.weight_decay      = weight_decay
        self.batch_size        = batch_size
        self.max_epochs        = max_epochs
        self.patience          = patience
        self.warmup_ratio      = warmup_ratio
        self.max_length        = max_length
        self.device            = device or get_device()
        self.random_state      = random_state

        self._model     = None
        self._tokenizer = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build(self, pretrained=True):
        """Construct tokeniser and model (backbone + head). Returns (tok, model).

        pretrained=False skips loading HuggingFace weights — use when the caller
        is about to overwrite them via load_state_dict (avoids loading ~1GB twice).
        """
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel, AutoConfig

        # local_files_only=True — model is already cached; skip ALL Hub network
        # calls (version checks, telemetry) that can hang on slow/no network.
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, local_files_only=True)
        if pretrained:
            backbone = AutoModel.from_pretrained(
                self.model_name, local_files_only=True)
        else:
            # Build architecture from config only — no weight download/copy.
            cfg      = AutoConfig.from_pretrained(
                self.model_name, local_files_only=True)
            backbone = AutoModel.from_config(cfg)

        # Freeze entire backbone first
        for param in backbone.parameters():
            param.requires_grad = False

        # Selectively unfreeze top N transformer encoder layers
        if self.n_unfreeze_layers > 0:
            for layer in backbone.encoder.layer[-self.n_unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        hidden_size = backbone.config.hidden_size  # 768 for base

        class _Model(nn.Module):
            def __init__(self_, backbone, dropout, hidden_size):
                super().__init__()
                self_.backbone   = backbone
                self_.dropout    = nn.Dropout(dropout)
                self_.head       = nn.Linear(hidden_size, 1)
                nn.init.xavier_uniform_(self_.head.weight)
                nn.init.zeros_(self_.head.bias)

            def forward(self_, input_ids, attention_mask):
                out    = self_.backbone(input_ids=input_ids, attention_mask=attention_mask)
                cls    = out.last_hidden_state[:, 0, :]            # (B, H) — [CLS] token
                return self_.head(self_.dropout(cls)).squeeze(-1)  # (B,) logits

        model = _Model(backbone, self.dropout, hidden_size)
        return tokenizer, model

    def _make_loader(self, texts, labels=None, shuffle=False, override_batch=None):
        """Return a DataLoader that tokenises lazily per batch (no upfront blocking call)."""
        import torch
        from torch.utils.data import DataLoader

        tokenizer  = self._tokenizer
        max_length = self.max_length

        def collate_fn(batch):
            raw = [b["text"] for b in batch]
            enc = tokenizer(raw, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            out = {"input_ids": enc["input_ids"],
                   "attention_mask": enc["attention_mask"]}
            if "labels" in batch[0]:
                out["labels"] = torch.tensor(
                    [b["labels"] for b in batch], dtype=torch.float32
                )
            return out

        bs = override_batch if override_batch is not None else (
            self.batch_size if shuffle else self.batch_size * 2
        )
        ds = _PhishBERTDataset(texts, labels)
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=0,   # macOS "spawn" mode deadlocks with workers > 0 on MPS
            collate_fn=collate_fn,
        )

    def _raw_logits(self, loader):
        """Run the model in eval mode and return concatenated logits."""
        import torch
        self._model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in loader:
                logits = self._model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                all_logits.append(logits.cpu())
        return torch.cat(all_logits)

    def _val_auc(self, loader, y_true):
        from sklearn.metrics import roc_auc_score
        import torch
        probs = torch.sigmoid(self._raw_logits(loader)).numpy()
        return roc_auc_score(y_true, probs)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, texts, y, val_texts=None, val_y=None):
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        torch.manual_seed(self.random_state)

        n_gpu = get_n_gpu()
        print(
            f"[bert] Building PhishBERT "
            f"(unfreeze_layers={self.n_unfreeze_layers}, device={self.device}"
            + (f", {n_gpu} GPUs" if n_gpu > 1 else "") + ")…"
        )
        self._tokenizer, self._model = self._build()
        self._model = self._model.to(self.device)
        if self.device == "cuda" and n_gpu > 1:
            print(f"[bert] Wrapping in DataParallel ({n_gpu} GPUs)")
            self._model = nn.DataParallel(self._model)

        # Scale batch size across GPUs so each GPU sees self.batch_size samples
        effective_batch = self.batch_size * max(1, n_gpu)

        n_val = len(val_texts) if val_texts is not None else 0
        print(f"[bert] Building data loaders ({len(texts)} train, {n_val} val)…")
        train_loader = self._make_loader(texts, y, shuffle=True,
                                         override_batch=effective_batch)
        val_loader   = (
            self._make_loader(val_texts, val_y) if val_texts is not None else None
        )

        # Class-imbalance weighting
        n_neg      = int((np.array(y) == 0).sum())
        n_pos      = int((np.array(y) == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Separate learning-rate groups: higher LR for the fresh head
        head_params     = list(self._model.head.parameters())
        head_ids        = {id(p) for p in head_params}
        backbone_params = [p for p in self._model.parameters()
                           if p.requires_grad and id(p) not in head_ids]

        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": self.lr},
                {"params": head_params,     "lr": self.head_lr},
            ],
            weight_decay=self.weight_decay,
        )

        total_steps   = len(train_loader) * self.max_epochs
        warmup_steps  = int(total_steps * self.warmup_ratio)
        scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        best_val_auc = 0.0
        best_state   = None
        no_improve   = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            total_loss = 0.0
            pbar = tqdm(train_loader, desc=f"[bert] Epoch {epoch + 1}/{self.max_epochs}", leave=True)
            for batch in pbar:
                optimizer.zero_grad()
                logits = self._model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                loss = criterion(logits, batch["labels"].to(self.device))
                loss.backward()
                _params = (self._model.module if hasattr(self._model, "module")
                           else self._model).parameters()
                nn.utils.clip_grad_norm_(_params, 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Flush MPS command buffer and release fragmented memory before
            # val inference and the next epoch — without this, MPS can stall
            # indefinitely at the epoch boundary on Apple Silicon.
            if str(self.device) == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()

            avg_loss = total_loss / len(train_loader)

            if val_loader is not None:
                val_auc = self._val_auc(val_loader, val_y)
                tqdm.write(
                    f"[bert]   Epoch {epoch + 1}/{self.max_epochs}  "
                    f"loss={avg_loss:.4f}  val_AUC={val_auc:.4f}"
                )
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    _raw = self._model.module if hasattr(self._model, "module") else self._model
                    best_state   = {k: v.detach().cpu() for k, v in _raw.state_dict().items()}
                    no_improve   = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        tqdm.write(f"[bert]   Early stopping at epoch {epoch + 1}")
                        break
            else:
                tqdm.write(f"[bert]   Epoch {epoch + 1}/{self.max_epochs}  loss={avg_loss:.4f}")

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[bert] Restored best checkpoint (val_AUC={best_val_auc:.4f})")

        return self

    def predict_proba(self, texts):
        """Return (N, 2) array [P_benign, P_phish] — sklearn convention."""
        import torch
        loader = self._make_loader(texts)
        probs  = torch.sigmoid(self._raw_logits(loader)).numpy()
        return np.column_stack([1 - probs, probs])

    def save(self, path):
        import torch
        # Unwrap DataParallel before saving so the checkpoint is device-agnostic
        raw_model = self._model.module if hasattr(self._model, "module") else self._model
        torch.save(
            {
                # Always save on CPU so the checkpoint loads on any device
                # without map_location bugs (MPS tensors hang on macOS ARM64).
                "model_state": {k: v.detach().cpu() for k, v in raw_model.state_dict().items()},
                "config": {
                    "model_name":        self.model_name,
                    "n_unfreeze_layers": self.n_unfreeze_layers,
                    "dropout":           self.dropout,
                    "max_length":        self.max_length,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        import gc
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoConfig, AutoModel

        # 1. Force single-threading to prevent macOS thread contention
        #    during the weight copy phase.
        torch.set_num_threads(1)

        print("[bert.load] reading checkpoint …", flush=True)
        # 2. mmap=False avoids filesystem-level hangs on some macOS versions.
        data = torch.load(path, map_location="cpu", weights_only=False,
                          mmap=False)
        cfg  = data["config"]

        obj = cls(
            model_name=cfg["model_name"],
            n_unfreeze_layers=cfg["n_unfreeze_layers"],
            dropout=cfg["dropout"],
            max_length=cfg["max_length"],
            device=device,
        )

        print("[bert.load] tokenizer …", flush=True)
        obj._tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name"], local_files_only=True)

        print("[bert.load] backbone architecture …", flush=True)
        hf_cfg   = AutoConfig.from_pretrained(
            cfg["model_name"], local_files_only=True)
        backbone = AutoModel.from_config(hf_cfg)

        hidden_size = backbone.config.hidden_size

        class _M(nn.Module):
            def __init__(s, backbone, dropout, hidden_size):
                super().__init__()
                s.backbone = backbone
                s.dropout  = nn.Dropout(dropout)
                s.head     = nn.Linear(hidden_size, 1)
            def forward(s, input_ids, attention_mask):
                out = s.backbone(input_ids=input_ids, attention_mask=attention_mask)
                cls_tok = out.last_hidden_state[:, 0, :]
                return s.head(s.dropout(cls_tok)).squeeze(-1)

        obj._model = _M(backbone, obj.dropout, hidden_size)

        # 3. Strip "module." prefix added by DataParallel saves.
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in data["model_state"].items()
        }

        print("[bert.load] loading weights …", flush=True)
        obj._model.load_state_dict(state)

        # 4. Free the checkpoint dict immediately to avoid RAM spike freezing the OS.
        del data, state
        gc.collect()

        print("[bert.load] to device / eval …", flush=True)
        obj._model = obj._model.to(obj.device)
        obj._model.eval()
        print("[bert.load] ready.", flush=True)
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Transformer / embedding utilities (used by train.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_transformer(device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = device or get_device()
    n_gpu  = get_n_gpu()
    print(f"[model] Loading {TRANSFORMER_NAME} on {device}...")
    tokenizer   = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    transformer = AutoModel.from_pretrained(TRANSFORMER_NAME).to(device)
    if device == "cuda" and n_gpu > 1:
        print(f"[model] Wrapping transformer in DataParallel ({n_gpu} GPUs)")
        transformer = torch.nn.DataParallel(transformer)
    transformer.eval()
    return tokenizer, transformer, device


def embed_texts(tokenizer, transformer, device, texts,
                batch_size=EMB_BATCH_SIZE, desc="Embedding"):
    import torch
    # Scale batch size across all available GPUs
    n_gpu          = get_n_gpu()
    effective_bs   = batch_size * max(1, n_gpu)
    all_embs = []
    for i in tqdm(range(0, len(texts), effective_bs), desc=desc, leave=False):
        batch = texts[i:i + effective_bs]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        mask      = enc["attention_mask"].to(device)
        with torch.no_grad():
            out  = transformer(input_ids, attention_mask=mask)
            # DataParallel may return a plain tuple; handle both cases
            last = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
            m    = mask.unsqueeze(-1).expand(last.size()).float()
            emb  = (last * m).sum(1) / m.sum(1).clamp(min=1e-9)
            all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)
