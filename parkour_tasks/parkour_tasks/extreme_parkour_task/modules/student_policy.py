from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
from torch import nn, Tensor

from .tokenizers.proprio_encoder import ProprioEncoder
from .tokenizers.depth_encoder import DepthEncoder
from .encoders.fusion_transformer import MultiModalFusionTransformer
from .temperal.txl import TransformerXLTemporal
from .actionheads.joint_action_head import JointPoseActionHead


@dataclass
class TXLStudentConfig:
    """Configuration for the TXL-based student policy."""

    # Proprio encoder
    proprio_state_dim: int
    proprio_hist_len: int = 3
    proprio_token_dim: int = 128

    # Depth encoder
    depth_in_frames: int = 4
    depth_in_size: int = 64
    depth_token_dim: int = 128
    depth_grid_size: int = 4

    # Fusion transformer
    fusion_num_layers: int = 2
    fusion_num_heads: int = 4
    fusion_mlp_ratio: float = 2.0
    fusion_dropout: float = 0.1
    fusion_attn_dropout: float = 0.1

    # Temporal TXL
    txl_d_model: int = 128
    txl_n_layer: int = 4
    txl_n_head: int = 4
    txl_d_inner: int = 256
    txl_mem_len: int = 64
    txl_dropout: float = 0.1
    txl_attn_dropout: float = 0.1

    # Action head
    action_dim: int = 12
    action_hidden_dims: Tuple[int, ...] = (256, 256)
    tanh_output: bool = False
    action_scale: float = 1.0


class TXLStudentPolicy(nn.Module):
    """TXL-based student policy used for online DAgger with a trained teacher.

    High-level structure:
        proprio history  -> ProprioEncoder  -> [B, 1, D]
        depth frames     -> DepthEncoder    -> [B, N_vis, D]
                           -> MultiModalFusionTransformer -> fused pooled [B, D]
        collect T steps of fused latent -> TransformerXLTemporal -> [B, D]
        -> JointPoseActionHead -> Gaussian policy over joint poses
    """

    def __init__(self, cfg: TXLStudentConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Encoders
        self.proprio_encoder = ProprioEncoder(
            state_dim=cfg.proprio_state_dim,
            hist_len=cfg.proprio_hist_len,
            token_dim=cfg.proprio_token_dim,
        )
        self.depth_encoder = DepthEncoder(
            in_frames=cfg.depth_in_frames,
            in_size=cfg.depth_in_size,
            token_dim=cfg.depth_token_dim,
            grid_size=cfg.depth_grid_size,
        )
        if cfg.proprio_token_dim != cfg.depth_token_dim:
            raise ValueError(
                f"Token dims must match for fusion, got "
                f"{cfg.proprio_token_dim} and {cfg.depth_token_dim}."
            )
        self.fusion = MultiModalFusionTransformer(
            token_dim=cfg.proprio_token_dim,
            num_layers=cfg.fusion_num_layers,
            num_heads=cfg.fusion_num_heads,
            mlp_ratio=cfg.fusion_mlp_ratio,
            dropout=cfg.fusion_dropout,
            attn_dropout=cfg.fusion_attn_dropout,
        )

        # Temporal TXL
        self.temporal = TransformerXLTemporal(
            d_model=cfg.txl_d_model,
            n_layer=cfg.txl_n_layer,
            n_head=cfg.txl_n_head,
            d_inner=cfg.txl_d_inner,
            mem_len=cfg.txl_mem_len,
            dropout=cfg.txl_dropout,
            attn_dropout=cfg.txl_attn_dropout,
        )

        # Action head
        self.action_head = JointPoseActionHead(
            d_model=cfg.txl_d_model,
            action_dim=cfg.action_dim,
            hidden_dims=cfg.action_hidden_dims,
            tanh_output=cfg.tanh_output,
            action_scale=cfg.action_scale,
        )

        self._mems: Optional[list[Optional[Tensor]]] = None

    def reset(self, batch_size: int) -> None:
        """Reset TXL memories, called when environments reset."""
        self._mems = self.temporal.reset_mems(batch_size)

    def detach_mems(self) -> None:
        """Detach TXL memories between updates."""
        if self._mems is not None:
            self._mems = self.temporal.detach_mems(self._mems)

    def forward(
        self,
        proprio_hist: Tensor,
        depth_frames: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute student policy distribution for a single time step.

        Args:
            proprio_hist: [B, hist_len * state_dim]
            depth_frames: [B, C=in_frames, H, W]

        Returns:
            dict with keys: "mean", "log_std", "std", "dist"
        """
        if proprio_hist.dim() != 2:
            raise ValueError("proprio_hist must be [B, hist_len * state_dim].")
        if depth_frames.dim() != 4:
            raise ValueError("depth_frames must be [B, C, H, W].")

        batch_size = proprio_hist.size(0)
        if self._mems is None or (self._mems and self._mems[0].size(0) != batch_size):
            self.reset(batch_size)

        # Encode proprio into single token [B, 1, D]
        prop_tokens = self.proprio_encoder(proprio_hist)

        # Encode depth into visual tokens [B, N_vis, D]
        vis_tokens = self.depth_encoder(depth_frames)

        # Fuse tokens with transformer, use all_pooled as step latent [B, D]
        fused = self.fusion(prop_tokens, vis_tokens)
        step_latent = fused["all_pooled"].unsqueeze(1)  # [B, 1, D]

        # Temporal TXL over step latents
        temporal_out, self._mems = self.temporal(step_latent, mems=self._mems)
        step_with_history = temporal_out[:, -1, :]  # [B, D]

        # Gaussian action head
        return self.action_head(step_with_history)

    def forward_sequence(
        self,
        proprio_hist_seq: Tensor,
        depth_frames_seq: Tensor,
        mems: Optional[list[Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, list[Optional[Tensor]]]:
        """Process a sequence for TXL with truncated BPTT.

        Inputs: proprio_hist_seq [B, T, hist_len * state_dim], depth_frames_seq [B, T, C, H, W].
        Outputs: temporal_out [B, T, D]. Intended for sequence training with truncated BPTT
        (e.g., context length 128 as used in the Minecraft RL paper) while keeping forward for
        single-step online inference.
        """
        B, T, _ = proprio_hist_seq.shape
        _, _, C, H, W = depth_frames_seq.shape

        proprio_flat = proprio_hist_seq.reshape(B * T, -1)
        depth_flat = depth_frames_seq.reshape(B * T, C, H, W)

        prop_tokens = self.proprio_encoder(proprio_flat)
        vis_tokens = self.depth_encoder(depth_flat)

        fused = self.fusion(prop_tokens, vis_tokens)
        step_latent_flat = fused["all_pooled"]

        step_latent = step_latent_flat.view(B, T, -1)

        if mems is None:
            mems_in = self._mems
            if mems_in is None or (mems_in and mems_in[0].size(0) != B):
                self.reset(B)
                mems_in = self._mems
        else:
            mems_in = mems

        temporal_out, new_mems = self.temporal(step_latent, mems=mems_in)

        if mems is None:
            self._mems = new_mems

        return temporal_out, new_mems


def build_default_txl_student(
    proprio_state_dim: int,
    action_dim: int,
) -> TXLStudentPolicy:
    """Helper to build a default-config TXL student."""
    cfg = TXLStudentConfig(
        proprio_state_dim=proprio_state_dim,
        action_dim=action_dim,
    )
    return TXLStudentPolicy(cfg)
