from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from transformerxl.actionheads.joint_action_head import JointPoseActionHead
from transformerxl.encoders.fusion_transformer import MultiModalFusionTransformer
from transformerxl.temporal.txl import TransformerXLTemporal
from transformerxl.tokenizers.depth_encoder import DepthEncoder
from transformerxl.tokenizers.proprio_encoder import ProprioEncoder


class MultiModalStudentPolicy(nn.Module):
    """Full student policy combining tokenizers, fusion transformer, temporal TXL, and action head."""

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        camera_resolution: Tuple[int, int],
        prop_hist_len: int,
        depth_hist_len: int,
        fusion_cfg: Dict[str, object],
        temporal_cfg: Dict[str, object],
        action_head_cfg: Dict[str, object],
        token_dim: int = 128,
    ) -> None:
        super().__init__()
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        height, width = camera_resolution

        self.proprio_encoder = ProprioEncoder(
            state_dim=proprio_dim,
            hist_len=prop_hist_len,
            token_dim=token_dim,
            hidden_dims=fusion_cfg.get("prop_hidden_dims", (256, 256)),
            dropout=fusion_cfg.get("prop_dropout", 0.1),
        )
        self.depth_encoder = DepthEncoder(
            in_frames=depth_hist_len,
            in_size=max(height, width),
            token_dim=token_dim,
            grid_size=fusion_cfg.get("grid_size", 4),
            dropout=fusion_cfg.get("depth_dropout", 0.1),
        )
        self.fusion_transformer = MultiModalFusionTransformer(
            token_dim=token_dim,
            num_layers=fusion_cfg.get("num_layers", 2),
            num_heads=fusion_cfg.get("num_heads", 4),
            mlp_ratio=fusion_cfg.get("mlp_ratio", 2.0),
            dropout=fusion_cfg.get("dropout", 0.1),
            attn_dropout=fusion_cfg.get("attn_dropout", 0.1),
            add_modality_embed=True,
            norm_first=True,
        )
        self.temporal_model = TransformerXLTemporal(
            d_model=token_dim,
            n_layer=temporal_cfg.get("num_layers", 3),
            n_head=temporal_cfg.get("num_heads", 4),
            d_inner=temporal_cfg.get("d_inner", 256),
            mem_len=temporal_cfg.get("mem_len", 64),
            dropout=temporal_cfg.get("dropout", 0.1),
            attn_dropout=temporal_cfg.get("attn_dropout", 0.1),
            norm_first=temporal_cfg.get("norm_first", True),
            clamp_len=temporal_cfg.get("clamp_len", None),
            use_rel_pos=temporal_cfg.get("use_rel_pos", True),
        )
        self.action_head = JointPoseActionHead(
            d_model=token_dim,
            action_dim=action_dim,
            hidden_dims=action_head_cfg.get("hidden_dims", (256, 256)),
            tanh_output=action_head_cfg.get("tanh_output", False),  # 应该使用激活函数吗？教师模型tanh_encoder_output = False，会输出>1的action
            action_scale=action_head_cfg.get("action_scale", 1.0),
        )
        self.yaw_head = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(
        self,
        proprio_seq: Tensor,
        depth_seq: Tensor,
        full_dones: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Original simple forward (stateless).
        Args:
            proprio_seq: Tensor[B, S, prop_hist_len * proprio_dim]
            depth_seq: Tensor[B, S, depth_hist_len, H, W]
        """
        actions, _, _ = self.forward_with_mems(
            proprio_seq,
            depth_seq,
            mems=None,
            full_dones=full_dones,
        )
        return actions

    def forward_with_mems(
        self,
        proprio_seq: Tensor,
        depth_seq: Tensor,
        mems: Optional[List[Tensor]] = None,
        full_dones: Optional[Tensor] = None, # <--- 新增
    ):
        """
        Forward pass with segment recurrence memory support (TBPTT).

        Args:
            proprio_seq: Tensor[B, S, prop_hist_len * proprio_dim]
            depth_seq: Tensor[B, S, depth_hist_len, H, W]
            mems: Optional list of memory tensors from previous segment (should be detached)

        Returns:
            actions: Predicted action means of shape [B, S, action_dim]
            new_mems: List of new memory tensors for next segment
        """

        batch_size, seq_len, feat_dim = proprio_seq.shape
        prop_encoded = self.proprio_encoder(
            proprio_seq.reshape(batch_size * seq_len, feat_dim)
        )  # [B*S, prop_hist_len*proprio_dim]
        depth_encoded = self.depth_encoder(
            depth_seq.reshape(batch_size * seq_len, depth_seq.size(2), depth_seq.size(3), depth_seq.size(4))
        )  # [B*S, depth_hist_len, H, W]
        fused = self.fusion_transformer(prop_encoded, depth_encoded)
        fused_seq = fused["all_pooled"].reshape(batch_size, seq_len, -1)

        # Temporal modeling with memory
        temporal_out, new_mems = self.temporal_model(
            fused_seq,
            mems=mems,
            return_mems=True,
            full_dones=full_dones
        )
        actions = self.action_head.forward_sequence(temporal_out)["mean"]
        raw_yaws = self.yaw_head(temporal_out)
        predicted_yaws = 1.5 * torch.tanh(raw_yaws)
        return actions, predicted_yaws, new_mems

    def forward_with_mems_rl(
        self,
        proprio_seq: Tensor,
        depth_seq: Tensor,
        old_actions: Optional[Tensor] = None,
        mems: Optional[List[Tensor]] = None,
        full_dones: Optional[Tensor] = None, # <--- 新增
    ):
        """
        Returns
        -------
        actions   : [B, S, A]  sampled (rollout) or ``old_actions`` (update)
        log_probs : [B, S]     sum of log-prob over action dimensions
        entropy   : [B, S]     sum of entropy  over action dimensions
        pred_yaws : [B, S, 2]  auxiliary heading prediction
        new_mems  : List[Tensor]
        """
        B, S, F = proprio_seq.shape

        # ── encode ──────────────────────────────────────────────────────────
        prop_encoded = self.proprio_encoder(proprio_seq.reshape(B * S, F))
        depth_encoded = self.depth_encoder(
            depth_seq.reshape(B * S,
                              depth_seq.size(2), depth_seq.size(3), depth_seq.size(4))
        )
        fused = self.fusion_transformer(prop_encoded, depth_encoded)
        fused_seq = fused["all_pooled"].reshape(B, S, -1)

        # ── temporal model ───────────────────────────────────────────────────
        temporal_out, new_mems = self.temporal_model(
            fused_seq, mems=mems, return_mems=True, full_dones=full_dones
        )

        # ── action distribution ──────────────────────────────────────────────
        # JointPoseActionHead.forward_sequence() already builds Normal(mean, std)
        action_out = self.action_head.forward_sequence(temporal_out)
        dist = action_out["dist"]                     # Normal [B, S, A]

        actions = dist.sample() if old_actions is None else old_actions
        log_probs = dist.log_prob(actions).sum(-1)         # [B, S]
        entropy = dist.entropy().sum(-1)                 # [B, S]

        # ── auxiliary yaw head ───────────────────────────────────────────────
        pred_yaws = 1.5 * torch.tanh(self.yaw_head(temporal_out))  # [B, S, 2]

        return actions, log_probs, entropy, pred_yaws, new_mems
