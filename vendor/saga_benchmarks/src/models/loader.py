"""
src/models/loader.py

Sequential GPU offloading strategy:
  - Models are instantiated with weights on CPU.
  - load_to_gpu() moves one model to cuda:0 for encoding.
  - offload_to_cpu() moves it back and clears the CUDA cache.
  - Only ONE base model occupies GPU VRAM at a time during encoding.
  - The Meta-Model lives permanently on cuda:1.

This pattern avoids OOM when 3 frozen base models + projectors + optimizer
states must coexist during alignment training.

Tokenizer heterogeneity note:
  The three models use BPE (Qwen), SentencePiece (Gemma), and a custom
  tokenizer (SmolLM). Mean-pooling is performed over the model's own
  attention_mask, so padding contamination is correctly excluded regardless
  of tokenizer vocabulary or granularity differences. The projectors are
  then trained contrastively to align the resulting pooled vectors.
"""
from __future__ import annotations
import yaml
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from typing import Dict, List, Optional


class FrozenModelWrapper(nn.Module):
    """
    Lazy-loading frozen causal LM with sequential GPU offloading.

    Public API:
        load_to_gpu()          → self
        offload_to_cpu()       → None
        encode(prompts)        → Tensor[B, hidden_dim]  (CPU, float32)
        generate(prompts)      → List[str]
    """

    def __init__(self, model_cfg: dict, encoding_device: str = "cuda:0"):
        super().__init__()
        self.cfg = model_cfg
        self.model_id = model_cfg["id"]
        self.hf_name = model_cfg["hf_name"]
        self.commit = model_cfg.get("commit", "main")
        self.hidden_dim = model_cfg["hidden_dim"]
        self.encoding_device = encoding_device

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.dtype = dtype_map[model_cfg.get("dtype", "bfloat16")]

        # Tokenizer loads to CPU immediately (negligible memory)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_name, revision=self.commit, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"   # consistent for causal LMs

        self._model: Optional[AutoModelForCausalLM] = None

    def _ensure_loaded(self):
        if self._model is None:
            print(f"  [loader] Loading {self.model_id} weights to CPU...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.hf_name,
                revision=self.commit,
                torch_dtype=self.dtype,
                device_map="cpu",
                trust_remote_code=True,
                output_hidden_states=True,
            )
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)

    def load_to_gpu(self) -> "FrozenModelWrapper":
        """Move model weights to encoding_device. Call before encode()."""
        self._ensure_loaded()
        self._model.to(self.encoding_device)
        return self

    def offload_to_cpu(self) -> None:
        """Move model back to CPU and release CUDA cache. Call after encode()."""
        if self._model is not None:
            self._model.to("cpu")
        torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(self, prompts: List[str], max_length: int = 256) -> torch.Tensor:
        """
        Returns Tensor[B, hidden_dim] on CPU (float32).
        Mean-pools last hidden state over non-padding positions.
        The model's own attention_mask is used, so this is correct
        regardless of tokenizer type (BPE / SentencePiece / custom).
        """
        assert self._model is not None and \
               next(self._model.parameters()).device.type == "cuda", \
            "Call load_to_gpu() before encode()"

        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        input_ids = enc["input_ids"].to(self.encoding_device)
        attention_mask = enc["attention_mask"].to(self.encoding_device)

        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1].float()   # [B, seq, dim]
        mask = attention_mask.unsqueeze(-1).float()        # [B, seq, 1]
        summed = (last_hidden * mask).sum(dim=1)           # [B, dim]
        counts = mask.sum(dim=1).clamp(min=1e-9)           # [B, 1]
        return (summed / counts).cpu()                     # [B, dim]

    @torch.no_grad()
    def generate(self, prompts: List[str], max_new_tokens: int = 256,
                 **kwargs) -> List[str]:
        """Generate text. Model must be on GPU (call load_to_gpu() first)."""
        assert self._model is not None
        enc = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=256,
        )
        input_ids = enc["input_ids"].to(self.encoding_device)
        attention_mask = enc["attention_mask"].to(self.encoding_device)
        past_key_values = DynamicCache()
        out = self._model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False, **kwargs,
        )
        new_tokens = out[:, input_ids.shape[1]:]
        return self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def load_all_models(
    config_path: str = "configs/models.yaml",
    encoding_device: str = "cuda:0",
) -> Dict[str, FrozenModelWrapper]:
    """
    Instantiate all base model wrappers (CPU-resident until encode() is called).
    Validates that commit hashes have been filled in by the human operator.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for m in cfg["base_models"]:
        assert "FILL_FROM" not in str(m.get("commit", "")), (
            f"Commit hash not filled for '{m['id']}'. "
            f"Complete Step 0.A.7 first."
        )

    models = {}
    for model_cfg in cfg["base_models"]:
        models[model_cfg["id"]] = FrozenModelWrapper(model_cfg, encoding_device)
        print(f"  [loader] Registered '{model_cfg['id']}' (lazy CPU)")
    return models


def sequential_encode(
    models: Dict[str, FrozenModelWrapper],
    prompts: List[str],
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    Encode prompts through all models sequentially.
    Each model is GPU-loaded, encoded, then offloaded before the next.
    Returns {model_id: Tensor[B, hidden_dim]} all on CPU.
    """
    embeddings = {}
    for model_id, wrapper in models.items():
        wrapper.load_to_gpu()
        embeddings[model_id] = wrapper.encode(prompts, max_length=max_length)
        wrapper.offload_to_cpu()
    return embeddings
