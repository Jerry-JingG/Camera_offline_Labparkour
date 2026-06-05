from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: type[nn.Module] = nn.ELU,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=5**0.5)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DepthCNNEncoder(nn.Module):
    """Lightweight CNN encoder for stacked depth frames."""

    def __init__(
        self,
        in_frames: int,
        out_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_frames <= 0:
            raise ValueError("in_frames must be positive.")

        self.cnn = nn.Sequential(
            nn.Conv2d(in_frames, 32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(
            nn.Linear(64, out_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=5**0.5)
            nn.init.zeros_(module.bias)

    def forward(self, depth_hist: Tensor) -> Tensor:
        if depth_hist.ndim != 4:
            raise ValueError(f"Expected depth history [B, T, H, W], got {tuple(depth_hist.shape)}.")
        return self.proj(self.cnn(depth_hist))


class OPGRUEstimator(nn.Module):
    """Only-proprioception estimator branch."""

    def __init__(
        self,
        proprio_dim: int,
        prop_hist_len: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prop_encoder = MLP(
            proprio_dim * prop_hist_len,
            hidden_dims=(256, 128),
            out_dim=embed_dim,
            activation=nn.ELU,
            layer_norm=True,
        )
        self.gru = nn.GRUCell(embed_dim, hidden_dim)

    def forward(self, prop_hist_flat: Tensor, hidden: Tensor) -> Tensor:
        prop_feat = self.prop_encoder(prop_hist_flat)
        return self.gru(prop_feat, hidden)


class VPGRUEstimator(nn.Module):
    """Vision-plus-proprioception estimator branch."""

    def __init__(
        self,
        proprio_dim: int,
        prop_hist_len: int,
        depth_hist_len: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.prop_encoder = MLP(
            proprio_dim * prop_hist_len,
            hidden_dims=(256, 128),
            out_dim=embed_dim,
            activation=nn.ELU,
            layer_norm=True,
        )
        self.depth_encoder = DepthCNNEncoder(
            in_frames=depth_hist_len,
            out_dim=embed_dim,
            dropout=dropout,
        )
        self.fusion = MLP(
            2 * embed_dim,
            hidden_dims=(256,),
            out_dim=embed_dim,
            activation=nn.ELU,
            layer_norm=True,
        )
        self.gru = nn.GRUCell(embed_dim, hidden_dim)

    def forward(self, prop_hist_flat: Tensor, depth_hist: Tensor, hidden: Tensor) -> Tensor:
        prop_feat = self.prop_encoder(prop_hist_flat)
        depth_feat = self.depth_encoder(depth_hist)
        fused = self.fusion(torch.cat([prop_feat, depth_feat], dim=-1))
        return self.gru(fused, hidden)


class RENetBCPolicy(nn.Module):
    """RENet-style BC policy with OP/VP GRU branches and oracle switching."""

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        prop_hist_len: int = 10,
        depth_hist_len: int = 2,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        action_hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.hidden_dim = hidden_dim

        self.op_estimator = OPGRUEstimator(
            proprio_dim=proprio_dim,
            prop_hist_len=prop_hist_len,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
        self.vp_estimator = VPGRUEstimator(
            proprio_dim=proprio_dim,
            prop_hist_len=prop_hist_len,
            depth_hist_len=depth_hist_len,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        actor_input_dim = 2 * hidden_dim
        self.action_head = MLP(
            actor_input_dim,
            hidden_dims=action_hidden_dims,
            out_dim=action_dim,
            activation=nn.ELU,
            layer_norm=False,
        )

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, Tensor]:
        if device is None:
            device = next(self.parameters()).device
        return {
            "op": torch.zeros(batch_size, self.hidden_dim, device=device),
            "vp": torch.zeros(batch_size, self.hidden_dim, device=device),
        }

    @staticmethod
    def detach_hidden(hidden: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return {name: value.detach() for name, value in hidden.items()}

    @staticmethod
    def reset_hidden(hidden: Dict[str, Tensor], done_mask: Tensor) -> None:
        if not done_mask.any():
            return
        for value in hidden.values():
            value[done_mask] = 0.0

    def forward_step(
        self,
        prop_hist_flat: Tensor,
        depth_hist: Tensor,
        use_op_mask: Tensor,
        hidden: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor | Dict[str, Tensor]]:
        batch_size = prop_hist_flat.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch_size, prop_hist_flat.device)

        use_op_mask = use_op_mask.to(device=prop_hist_flat.device, dtype=torch.bool).view(batch_size, 1)
        op_hidden = self.op_estimator(prop_hist_flat, hidden["op"])
        vp_hidden = self.vp_estimator(prop_hist_flat, depth_hist, hidden["vp"])

        mask_f = use_op_mask.to(dtype=op_hidden.dtype)
        fused = torch.cat([op_hidden * mask_f, vp_hidden * (1.0 - mask_f)], dim=-1)
        actions = self.action_head(fused)

        return {
            "actions": actions,
            "hidden": {"op": op_hidden, "vp": vp_hidden},
            "op_hidden": op_hidden,
            "vp_hidden": vp_hidden,
            "fused": fused,
        }


class RENetOnlineRunner:
    """Maintains RENet histories and GRU states for vectorized environments."""

    def __init__(
        self,
        model: RENetBCPolicy,
        num_envs: int,
        proprio_dim: int,
        prop_hist_len: int,
        depth_hist_len: int,
        camera_resolution: Tuple[int, int],
        device: torch.device,
    ) -> None:
        self.model = model
        self.num_envs = num_envs
        self.proprio_dim = proprio_dim
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.camera_resolution = camera_resolution
        self.device = device

        self.prop_hist = torch.zeros(
            num_envs, prop_hist_len, proprio_dim, dtype=torch.float32, device=device
        )
        self.depth_hist = torch.zeros(
            num_envs, depth_hist_len, *camera_resolution, dtype=torch.float32, device=device
        )
        self.hidden = self.model.init_hidden(num_envs, device=device)

    def reset(self) -> None:
        self.prop_hist.zero_()
        self.depth_hist.zero_()
        self.hidden = self.model.init_hidden(self.num_envs, device=self.device)

    def reset_done(self, done_mask: Tensor) -> None:
        done_mask = done_mask.to(device=self.device, dtype=torch.bool).view(-1)
        if not done_mask.any():
            return
        self.prop_hist[done_mask] = 0.0
        self.depth_hist[done_mask] = 0.0
        self.model.reset_hidden(self.hidden, done_mask)

    def _prepare_depth(self, depth_image: Tensor) -> Tensor:
        depth_image = depth_image.to(self.device)
        if depth_image.dim() == 4 and depth_image.shape[1] == 1:
            depth_image = depth_image.squeeze(1)
        if depth_image.dim() != 3:
            raise ValueError(f"Expected depth image [N, H, W] or [N, 1, H, W], got {tuple(depth_image.shape)}.")
        return depth_image

    def forward_step(
        self,
        obs_prop: Tensor,
        depth_image: Tensor,
        use_op_mask: Tensor,
        prev_done: Optional[Tensor] = None,
    ) -> Dict[str, Tensor | Dict[str, Tensor]]:
        if prev_done is not None:
            self.reset_done(prev_done)

        obs_prop = obs_prop.to(self.device)
        depth_image = self._prepare_depth(depth_image)
        use_op_mask = use_op_mask.to(self.device, dtype=torch.bool).view(-1)

        self.prop_hist = torch.roll(self.prop_hist, shifts=-1, dims=1)
        self.depth_hist = torch.roll(self.depth_hist, shifts=-1, dims=1)
        self.prop_hist[:, -1, :] = obs_prop
        self.depth_hist[:, -1, :, :] = depth_image

        prop_input = self.prop_hist.reshape(self.num_envs, -1)
        output = self.model.forward_step(
            prop_hist_flat=prop_input,
            depth_hist=self.depth_hist,
            use_op_mask=use_op_mask,
            hidden=self.hidden,
        )
        self.hidden = self.model.detach_hidden(output["hidden"])  # type: ignore[arg-type]
        return output

    @torch.no_grad()
    def act(
        self,
        obs_prop: Tensor,
        depth_image: Tensor,
        use_op_mask: Tensor,
        prev_done: Optional[Tensor] = None,
    ) -> Tensor:
        output = self.forward_step(obs_prop, depth_image, use_op_mask, prev_done=prev_done)
        return output["actions"]  # type: ignore[return-value]
