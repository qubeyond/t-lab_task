"""Tier-2 агент: Qwen2.5-0.5B-Instruct, действие выбирается скорингом кандидатов."""

from __future__ import annotations

import numpy as np
import torch

from memory.methods import predict_action

_MODEL = None
_TOK = None
_MODEL_NAME = None


def get_llm(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu"):
    global _MODEL, _TOK, _MODEL_NAME

    if _MODEL is None or _MODEL_NAME != model_name:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        _MODEL.to(device)
        _MODEL.eval()
        _MODEL_NAME = model_name

    return _MODEL, _TOK


SYSTEM = (
    "You are a support-ticket routing assistant. Every ticket must be routed to exactly one "
    "of these categories: billing, technical, account, refund, security, feature_request. "
    "Below are examples from this agent's own past routing decisions, each marked as either "
    "confirmed correct or confirmed wrong (do not assume the wrong ones show the right answer, "
    "only that the shown category was wrong for that ticket). Use them, together with the new "
    "ticket, to choose the single best category."
)


def build_prompt(query_text: str, retrieved: list[tuple[str, str, int]]) -> str:
    lines = [SYSTEM, ""]

    for text, label, correct in retrieved:
        if correct:
            lines.append(f'Ticket: "{text}"\n-> confirmed correct category: {label}')
        else:
            lines.append(f'Ticket: "{text}"\n-> agent routed this to "{label}", which was WRONG.')
        lines.append("")

    lines.append(f'Ticket: "{query_text}"')
    lines.append("-> category:")
    return "\n".join(lines)


@torch.no_grad()
def score_labels(prompt: str, labels: list[str], model, tok, device: str = "cpu") -> np.ndarray:
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    prompt_out = model(prompt_ids, use_cache=True)
    base_past = prompt_out.past_key_values
    last_logp = torch.log_softmax(prompt_out.logits[0, -1], dim=-1)

    scores = np.zeros(len(labels))

    for i, label in enumerate(labels):
        label_ids = tok(" " + label, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        token_logps = [last_logp[label_ids[0, 0]]]

        if label_ids.shape[1] > 1:
            out = model(label_ids[:, :-1], past_key_values=base_past, use_cache=True)
            logp = torch.log_softmax(out.logits[0], dim=-1)

            for j in range(label_ids.shape[1] - 1):
                token_logps.append(logp[j, label_ids[0, j + 1]])

        scores[i] = torch.stack(token_logps).mean().item()

    return scores


class LLMRouterAgent:
    def __init__(
        self,
        retrieval_method: str,
        action_names: list[str],
        k: int = 1,
        topk: int = 5,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cpu",
    ):
        self.retrieval_method = retrieval_method
        self.action_names = action_names
        self.k = k
        self.topk = topk
        self.model, self.tok = get_llm(model_name, device)
        self.device = device

    def act(
        self,
        t_eval: int,
        texts: list[str],
        E: np.ndarray,
        actions: np.ndarray,
        correct01: np.ndarray,
        mem_idxs: np.ndarray,
        global_default: int = 0,
    ) -> int:
        n_actions = len(self.action_names)
        _, retrieved_idx = predict_action(
            self.retrieval_method,
            t_eval,
            E,
            actions,
            correct01,
            mem_idxs,
            k=self.k,
            topk=self.topk,
            n_actions=n_actions,
            global_default=global_default,
            return_retrieved=True,
        )

        retrieved = [(texts[j], self.action_names[actions[j]], int(correct01[j])) for j in retrieved_idx]
        prompt = build_prompt(texts[t_eval], retrieved)
        scores = score_labels(prompt, self.action_names, self.model, self.tok, self.device)
        return int(np.argmax(scores))
