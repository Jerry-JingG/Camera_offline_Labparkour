from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DepthEncoder(nn.Module):
    """Encode stacked egocentric depth frames into visual tokens."""

    def __init__(
        self,
        in_frames: int = 4,
        in_size: int = 64,
        token_dim: int = 128,
        grid_size: int = 4,
        add_2d_pos_embed: bool = True,
        dropout: float = 0.0,
        num_prop: int = 0,  # Number of proprioception features
    ) -> None:
        super().__init__()
        if in_frames <= 0:
            raise ValueError("in_frames must be positive.")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive.")
        if token_dim <= 0:
            raise ValueError("token_dim must be positive.")

        self.in_frames = in_frames
        self.in_size = in_size
        self.token_dim = token_dim
        self.grid_size = grid_size
        self.add_2d_pos_embed = add_2d_pos_embed
        # Store num_prop for proprio input (matching RecurrentDepthBackbone design)
        self.num_prop = num_prop if num_prop is not None else 0

        self.conv1 = nn.Conv2d(in_frames, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, token_dim, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.pool: Optional[nn.Module] = (
            None if grid_size == 4 else nn.AdaptiveAvgPool2d((grid_size, grid_size))
        )

        self.drop = nn.Dropout(dropout)
        self.pos_embed: Optional[nn.Parameter]
        if add_2d_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(token_dim, grid_size, grid_size)
            )
        else:
            self.register_parameter("pos_embed", None)

        # Yaw prediction head: predicts 2D yaw from visual features
        # Input: flattened visual features [B, grid_size*grid_size*token_dim] + optional proprio
        # Output: [B, 2] yaw prediction
        input_dim = grid_size * grid_size * token_dim + (num_prop if num_prop > 0 else 0)
        self.yaw_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),  # 2D yaw output
        )

        self._init_weights()

    @property
    def num_tokens(self) -> int:
        return self.grid_size * self.grid_size

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _check_input(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"Expected input of shape [B, C, H, W], got {x.shape}.")
        if x.shape[1] != self.in_frames:
            raise ValueError(
                f"Expected {self.in_frames} input channels, got {x.shape[1]}."
            )

    def _resize_if_needed(self, x: Tensor) -> Tensor:
        """Resize depth frames so that height/width are multiples of 2."""

        height, width = x.shape[-2], x.shape[-1]
        new_height = height + (height % 2)
        new_width = width + (width % 2)
        if new_height == height and new_width == width:
            return x
        return F.interpolate(
            x,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: Tensor, proprio: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x: Tensor of shape [B, in_frames, H, W].
            proprio: Optional Tensor of shape [B, num_prop].
                     IMPORTANT: delta_yaw (indices 6:8) should be zeroed before calling this method.

        Returns:
            Tensor of shape [B, grid_size * grid_size * token_dim + 2].
            Last 2 dimensions are yaw predictions.
        """
        self._check_input(x)
        x = self._resize_if_needed(x)
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = self.act(self.conv3(h))
        h = self.conv4(h)
        if self.pool is not None:
            h = self.pool(h)
        elif h.shape[-1] != self.grid_size or h.shape[-2] != self.grid_size:
            h = nn.functional.adaptive_avg_pool2d(h, (self.grid_size, self.grid_size))

        if self.pos_embed is not None:
            h = h + self.pos_embed.unsqueeze(0)

        tokens = h.flatten(2).transpose(1, 2).contiguous()  # [B, num_tokens, token_dim]
        tokens = self.drop(tokens)

        # Flatten visual features for yaw prediction
        visual_features_flat = tokens.flatten(1)  # [B, num_tokens * token_dim]

        # Concatenate with proprio if provided (matching RecurrentDepthBackbone design)
        if proprio is not None and self.num_prop > 0:
            yaw_input = torch.cat([visual_features_flat, proprio], dim=1)
        else:
            yaw_input = visual_features_flat

        # Predict yaw from combined features
        yaw_pred = self.yaw_head(yaw_input)  # [B, 2]

        # Concatenate tokens and yaw: [B, num_tokens, token_dim] -> [B, num_tokens * token_dim + 2]
        tokens_flat = tokens.flatten(1)  # [B, num_tokens * token_dim]
        output = torch.cat([tokens_flat, yaw_pred], dim=1)  # [B, num_tokens * token_dim + 2]

        if not torch.isfinite(output).all():
            raise ValueError("Non-finite values detected in encoder output.")
        return output


if __name__ == "__main__":
    torch.manual_seed(0)

    # Test without proprio (num_prop=0)
    encoder = DepthEncoder()
    batch = torch.randn(2, 4, 64, 64)
    out = encoder(batch)
    # Output: [B, grid_size*grid_size*token_dim + 2] = [2, 16*128 + 2] = [2, 2050]
    expected_shape = (2, 16 * 128 + 2)
    assert out.shape == expected_shape, f"Unexpected shape {out.shape}, expected {expected_shape}"
    assert torch.isfinite(out).all(), "Output contains non-finite values."

    # Test with proprio (num_prop=53)
    encoder_with_prop = DepthEncoder(num_prop=53)
    proprio = torch.randn(2, 53)
    out_with_prop = encoder_with_prop(batch, proprio)
    assert out_with_prop.shape == expected_shape, f"Unexpected shape {out_with_prop.shape}"
    assert torch.isfinite(out_with_prop).all(), "Output contains non-finite values."

    encoder_small = DepthEncoder(grid_size=2)
    out_small = encoder_small(batch)
    # Output: [B, 2*2*128 + 2] = [2, 514]
    expected_small = (2, 4 * 128 + 2)
    assert out_small.shape == expected_small, f"Unexpected shape {out_small.shape}, expected {expected_small}"
    assert torch.isfinite(out_small).all(), "Output contains non-finite values."
    print("DepthEncoder test passed.")
