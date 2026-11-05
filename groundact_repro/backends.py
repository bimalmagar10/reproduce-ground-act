"""Unified inference wrapper for Alpamayo-R1 (`alpamayo_r1`) and Alpamayo 1.5 (`alpamayo1_5`).

Two generation modes, matching the "Mode" column of GroundAct Table 1:
  continuous  chain-of-causation (CoC) rollout by the VLM, then the flow-matching action expert
              (the released inference path, `sample_trajectories_from_data_with_vlm_rollout`).
  discrete    CoC rollout, then 128 discrete action tokens from the VLM vocabulary head, decoded
              with the checkpoint's DiscreteTrajectoryTokenizer (unicycle accel/curvature).

Six candidates per sample = six independent VLM rollouts (each with its own CoC), as in the
Alpamayo evaluation code the paper follows.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import einops
import numpy as np
import torch

from groundact_repro.discrete import DiscreteTrajectoryLogitsProcessor, count_valid_traj_tokens

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    pred_xyz: np.ndarray  # (S, 64, 3) ego frame at t0
    pred_rot: np.ndarray  # (S, 64, 3, 3)
    cot: list[str]
    mode: str
    n_traj_tokens: list[int] = field(default_factory=list)  # discrete mode only
    base_xyz: np.ndarray | None = None  # continuous+correction: the uncorrected plan
    extra: dict = field(default_factory=dict)


def detect_family(model_path: str) -> str:
    cfg = json.loads((Path(model_path) / "config.json").read_text())
    mt = cfg.get("model_type", "")
    if mt == "alpamayo_r1":
        return "r1"
    if mt == "alpamayo1_5":
        return "1.5"
    raise ValueError(f"unknown Alpamayo model_type {mt!r} in {model_path}")


class AlpamayoBackend:
    def __init__(self, model_path: str, family: str | None = None, attn: str = "flash_attention_2",
                 dtype=torch.bfloat16, device: str = "cuda", cuda_graph: bool = False):
        self.model_path = model_path
        self.family = family or detect_family(model_path)
        self.device = device
        if self.family == "r1":
            from alpamayo_r1 import helper
            from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1 as Cls
            from alpamayo_r1.models.token_utils import extract_text_tokens, extract_traj_tokens
        else:
            from alpamayo1_5 import helper
            from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5 as Cls
            from alpamayo1_5.models.token_utils import extract_text_tokens, extract_traj_tokens
        self.helper = helper
        self._extract_text = extract_text_tokens
        self._extract_traj = extract_traj_tokens
        config = Cls.config_class.from_pretrained(model_path)
        config.attn_implementation = attn  # README-documented switch (sdpa fallback)
        self.model = Cls.from_pretrained(model_path, config=config, dtype=dtype).to(device).eval()
        self.processor = helper.get_processor(self.model.tokenizer)
        self.tokenizer = self.model.tokenizer
        if cuda_graph:
            self.model.enable_diffusion_expert_cuda_graph(max_batch_size=16, max_graphs=4)
        tok = self.tokenizer
        self.ids = {
            "traj_future_start": tok.convert_tokens_to_ids("<|traj_future_start|>"),
            "traj_future_end": tok.convert_tokens_to_ids("<|traj_future_end|>"),
            "eos": tok.eos_token_id,
            "pad": tok.pad_token_id,
        }

    # ------------------------------------------------------------------ inputs
    def build_inputs(self, data: dict) -> dict:
        frames = data["image_frames"].flatten(0, 1)
        if self.family == "r1":
            messages = self.helper.create_message(frames)
        else:
            messages = self.helper.create_message(frames=frames, camera_indices=data["camera_indices"])
        tokenized = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, continue_final_message=True,
            return_dict=True, return_tensors="pt")
        inputs = {"tokenized_data": tokenized, "ego_history_xyz": data["ego_history_xyz"],
                  "ego_history_rot": data["ego_history_rot"]}
        return self.helper.to_device(inputs, self.device)

    # ------------------------------------------------------------------ generation
    @staticmethod
    def _fresh(inputs: dict) -> dict:
        """Shallow copy: the R1 rollout pops ``input_ids`` from the dict it is given."""
        out = dict(inputs)
        out["tokenized_data"] = dict(inputs["tokenized_data"])
        return out

    def generate(self, inputs: dict, mode: str = "continuous", num_candidates: int = 6,
                 top_p: float = 0.98, temperature: float = 0.6, max_cot_tokens: int = 256,
                 seed: int | None = 42, corrector=None) -> GenerationResult:
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if mode == "continuous":
                return self._generate_continuous(inputs, num_candidates, top_p, temperature,
                                                 max_cot_tokens, corrector)
            if mode == "discrete":
                return self._generate_discrete(inputs, num_candidates, top_p, temperature,
                                               max_cot_tokens)
        raise ValueError(mode)

    def _generate_continuous(self, inputs, S, top_p, temperature, max_cot_tokens, corrector):
        kwargs = dict(data=self._fresh(inputs), top_p=top_p, temperature=temperature, num_traj_samples=S,
                      max_generation_length=max_cot_tokens, return_extra=True)
        if corrector is None:
            pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(**kwargs)
            base = None
            cextra = {}
        else:
            from groundact_repro.groundact.correction import corrected_sampling
            with corrected_sampling(self.model, corrector, inputs) as handle:
                pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_vlm_rollout(**kwargs)
            base = handle.base_xyz
            cextra = handle.info
        cot = [str(c) for c in np.asarray(extra["cot"]).reshape(-1)]
        return GenerationResult(pred_xyz[0, 0].float().cpu().numpy(), pred_rot[0, 0].float().cpu().numpy(),
                                cot, "continuous", base_xyz=base, extra=cextra)

    def _generate_discrete(self, inputs, S, top_p, temperature, max_cot_tokens):
        model = self.model
        data = self._fresh(inputs)
        tokenized = data["tokenized_data"]
        input_ids = tokenized.pop("input_ids")
        hist_xyz, hist_rot = data["ego_history_xyz"], data["ego_history_rot"]
        input_ids = model.fuse_traj_tokens(input_ids, {"ego_history_xyz": hist_xyz,
                                                       "ego_history_rot": hist_rot})
        n_traj = model.config.tokens_per_future_traj  # 128
        fut_vocab = model.traj_tokenizer.vocab_size  # 3000
        proc = DiscreteTrajectoryLogitsProcessor(
            prompt_len=input_ids.shape[1], traj_future_start_id=self.ids["traj_future_start"],
            traj_future_end_id=self.ids["traj_future_end"],
            traj_token_start_idx=model.config.traj_token_start_idx,
            traj_vocab_size=model.config.traj_vocab_size, future_vocab_size=fut_vocab,
            tokens_per_future_traj=n_traj, eos_token_id=self.ids["eos"])
        gc = copy.deepcopy(model.vlm.generation_config)
        gc.do_sample, gc.top_p, gc.temperature, gc.top_k = True, top_p, temperature, None
        gc.num_return_sequences = S
        gc.max_new_tokens = max_cot_tokens + n_traj + 2
        gc.return_dict_in_generate = True
        gc.output_logits = False
        gc.pad_token_id = self.ids["pad"]
        gc.eos_token_id = [self.ids["traj_future_end"], self.ids["eos"]]
        from transformers import LogitsProcessorList
        out = model.vlm.generate(input_ids=input_ids, generation_config=gc,
                                 logits_processor=LogitsProcessorList([proc]), **tokenized)
        seqs = out.sequences
        traj_tokens = self._extract_traj(seqs, model.special_token_ids, n_traj,
                                         model.future_token_start_idx, fut_vocab)  # (S, 128)
        n_valid = count_valid_traj_tokens(seqs, input_ids.shape[1], self.ids["traj_future_start"],
                                          self.ids["traj_future_end"], model.future_token_start_idx,
                                          model.future_token_start_idx + fut_vocab)
        hist_xyz_rep = einops.repeat(hist_xyz[:, -1], "b ... -> (b n) ...", n=S).float()
        hist_rot_rep = einops.repeat(hist_rot[:, -1], "b ... -> (b n) ...", n=S).float()
        with torch.autocast("cuda", enabled=False):
            fut_xyz, fut_rot, _ = model.traj_tokenizer.decode(hist_xyz_rep, hist_rot_rep, traj_tokens)
        cot = self._extract_text(self.tokenizer, seqs)["cot"]
        return GenerationResult(fut_xyz.float().cpu().numpy(), fut_rot.float().cpu().numpy(),
                                [str(c) for c in cot], "discrete", n_traj_tokens=n_valid.tolist())
