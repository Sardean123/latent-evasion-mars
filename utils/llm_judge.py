"""Constrained LLM-judge scoring: let the judge reason, then read the score from logits.

The failure this exists to prevent: if you let a judge free-generate and then regex a
"Rating: X" line out of the text, the judge decides the output format. Our Llama-3-8B
self-judge emits 1.5 for ~86% of items and never once emits 0 or 2, so a nominal 0/1/2 rubric
silently degrades into a near-constant, plus ~7% unparseable responses that get dropped.

Instead: generate the explanation, append "Rating: ", and take an argmax over the logits of the
label tokens at that one position. The score is always on-rubric, nothing is unparseable, and
`captured_mass` reports how much of the next-token distribution actually landed on the label
tokens -- a low value means the judge wanted to say something else and the score is not
trustworthy.

Tokenizer gotcha this handles: in the Llama-3 vocabulary " 0" encodes to [220, 15] (space, then
digit), and that leading space token is shared by every label -- scoring on it makes all labels
identical. Only single-token encodings are used, and the forced prefix supplies the space.
"""
from typing import List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm


class ConstrainedJudge:
    def __init__(
        self,
        *,
        hf_id: Optional[str] = None,
        model=None,
        tokenizer=None,
        device: str = "cuda:0",
        labels: Sequence[str] = ("0", "1", "2"),
        max_reason_tokens: int = 160,
        batch_size: int = 8,
        force_prefix: str = "\nRating: ",
    ):
        """Either pass hf_id to load a judge, or pass an already-loaded model + tokenizer."""
        if hf_id is not None:
            import os
            from transformers import AutoModelForCausalLM, AutoTokenizer
            token = os.environ.get("HF_TOKEN")
            tokenizer = AutoTokenizer.from_pretrained(hf_id, token=token)
            model = AutoModelForCausalLM.from_pretrained(
                hf_id, token=token, dtype=torch.float16, device_map=device)
            model.eval()
        if model is None or tokenizer is None:
            raise ValueError("pass either hf_id, or both model and tokenizer")

        self.model = model
        self.tok = tokenizer
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.device = next(model.parameters()).device
        self.labels = list(labels)
        self.max_reason_tokens = max_reason_tokens
        self.batch_size = batch_size
        self.force_prefix = force_prefix
        self.label_ids = [self._single_token_ids(label) for label in self.labels]
        self.captured_mass: List[float] = []

    def _single_token_ids(self, label: str) -> List[int]:
        ids = set()
        for variant in (label, f" {label}"):
            enc = self.tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(enc[0])
        if not ids:
            raise ValueError(
                f"No single-token encoding for label {label!r}; pick labels that are single "
                f"tokens in this vocabulary, or score them by full-sequence likelihood instead.")
        return sorted(ids)

    def _chat(self, user_msg: str) -> str:
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_msg}], tokenize=False, add_generation_prompt=True)

    def score(self, user_msgs: List[str], desc: str = "judging") -> Tuple[List[int], List[str]]:
        """Returns (label indices, reasoning strings). Index i corresponds to self.labels[i]."""
        scores, reasons = [], []
        for start in tqdm(range(0, len(user_msgs), self.batch_size), desc=desc, leave=False):
            chunk = user_msgs[start:start + self.batch_size]
            texts = [self._chat(m) for m in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.device)

            if self.max_reason_tokens > 0:
                with torch.no_grad():
                    gen = self.model.generate(**enc, max_new_tokens=self.max_reason_tokens,
                                              do_sample=False, pad_token_id=self.tok.pad_token_id)
                plen = enc.input_ids.shape[1]
                chunk_reasons = [self.tok.decode(row[plen:], skip_special_tokens=True) for row in gen]
            else:
                chunk_reasons = ["" for _ in chunk]

            forced = [t + r + self.force_prefix for t, r in zip(texts, chunk_reasons)]
            enc2 = self.tok(forced, return_tensors="pt", padding=True,
                            add_special_tokens=False).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc2).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            for row, prow, reason in zip(logits, probs, chunk_reasons):
                cand = torch.tensor([row[ids].max() for ids in self.label_ids])
                scores.append(int(cand.argmax()))
                self.captured_mass.append(float(sum(prow[ids].sum() for ids in self.label_ids)))
                reasons.append(reason.strip())
        return scores, reasons

    def mean_captured_mass(self) -> float:
        return sum(self.captured_mass) / len(self.captured_mass) if self.captured_mass else 0.0
