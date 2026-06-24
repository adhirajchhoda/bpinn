
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class BPINNConfig:

    flow_channels: int = 2
    flow_hidden: int = 128
    depth_hidden: int = 64
    imu_channels: int = 6
    imu_hidden: int = 64
    imu_sequence_length: int = 200
    fusion_hidden: int = 256
    fusion_layers: int = 3
    dropout: float = 0.2
    use_flipout: bool = True
    prior_std: float = 1.0
    posterior_std_init: float = 0.1
    num_classes: int = 2
    charbonnier_epsilon: float = 0.001
    scale_reg_weight: float = 0.01

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

class FlipoutLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        posterior_std_init: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self._generator: Optional[torch.Generator] = None

        self.weight_mean = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_std = nn.Parameter(
            torch.full((out_features, in_features), math.log(posterior_std_init))
        )

        if bias:
            self.bias_mean = nn.Parameter(torch.empty(out_features))
            self.bias_log_std = nn.Parameter(
                torch.full((out_features,), math.log(posterior_std_init))
            )
        else:
            self.register_parameter("bias_mean", None)
            self.register_parameter("bias_log_std", None)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.weight_mean, mode="fan_in")
        if self.bias_mean is not None:
            nn.init.zeros_(self.bias_mean)

    def set_generator(self, generator: Optional[torch.Generator]) -> None:
        self._generator = generator

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        generator = self._generator

        sign_input = (
            torch.randint(
                0,
                2,
                (batch_size, self.in_features),
                device=x.device,
                generator=generator,
            ).to(dtype=x.dtype)
            * 2
            - 1
        )
        sign_output = (
            torch.randint(
                0,
                2,
                (batch_size, self.out_features),
                device=x.device,
                generator=generator,
            ).to(dtype=x.dtype)
            * 2
            - 1
        )

        output_mean = F.linear(x, self.weight_mean, self.bias_mean)

        weight_std = torch.exp(self.weight_log_std)
        weight_eps = torch.randn(
            self.weight_mean.shape,
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        weight_delta = weight_std * weight_eps

        perturbed_output = F.linear(x * sign_input, weight_delta, None)
        output_perturbation = perturbed_output * sign_output

        if self.bias_mean is not None:
            bias_std = torch.exp(self.bias_log_std)
            bias_eps = torch.randn(
                self.bias_mean.shape,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
            output_perturbation = output_perturbation + bias_std * bias_eps

        return output_mean + output_perturbation, self._compute_kl()

    def _compute_kl(self) -> torch.Tensor:
        weight_std = torch.exp(self.weight_log_std)
        prior_var = self.prior_std**2
        kl = 0.5 * torch.sum(
            (self.weight_mean**2 + weight_std**2) / prior_var
            - 1
            - 2 * self.weight_log_std
            + math.log(prior_var)
        )

        if self.bias_mean is not None:
            bias_std = torch.exp(self.bias_log_std)
            kl_bias = 0.5 * torch.sum(
                (self.bias_mean**2 + bias_std**2) / prior_var
                - 1
                - 2 * self.bias_log_std
                + math.log(prior_var)
            )
            kl = kl + kl_bias

        return kl

class FlowEncoder(nn.Module):

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Linear(256 * 4 * 4, hidden_dim)

    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(flow)
        x = self.pool(x)
        return self.proj(x.flatten(1))

class DepthEncoder(nn.Module):

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Linear(128 * 4 * 4, hidden_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        x = self.conv_layers(depth)
        x = self.pool(x)
        return self.proj(x.flatten(1))

class IMUEncoder(nn.Module):

    def __init__(
        self,
        input_channels: int = 6,
        hidden_dim: int = 64,
        sequence_length: int = 200,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.proj = nn.Linear(128 * 8, hidden_dim)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(imu)
        x = self.pool(x)
        return self.proj(x.flatten(1))

class BayesianFusionNetwork(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
        use_flipout: bool = True,
        prior_std: float = 1.0,
        posterior_std_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_flipout = use_flipout

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            if use_flipout:
                layers.append(
                    FlipoutLinear(
                        in_dim,
                        hidden_dim,
                        prior_std=prior_std,
                        posterior_std_init=posterior_std_init,
                    )
                )
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.layers = nn.ModuleList(layers)
        if use_flipout:
            self.output_layer = FlipoutLinear(
                hidden_dim,
                num_classes,
                prior_std=prior_std,
                posterior_std_init=posterior_std_init,
            )
        else:
            self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        total_kl = torch.tensor(0.0, device=x.device)

        for layer in self.layers:
            if isinstance(layer, FlipoutLinear):
                x, kl = layer(x)
                total_kl = total_kl + kl
            else:
                x = layer(x)

        if isinstance(self.output_layer, FlipoutLinear):
            logits, kl = self.output_layer(x)
            total_kl = total_kl + kl
        else:
            logits = self.output_layer(x)

        return logits, total_kl

class TranslationPredictor(nn.Module):

    def __init__(self, imu_dim: int = 64, depth_dim: int = 64) -> None:
        super().__init__()
        self.velocity_head = nn.Sequential(
            nn.Linear(imu_dim + depth_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    def forward(
        self, imu_features: torch.Tensor, depth_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.velocity_head(torch.cat([imu_features, depth_features], dim=-1))
        v = params[:, :3]
        scale = F.softplus(params[:, 3:4])
        return v, scale

def compute_translational_flow_v16(
    v: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:

    batch, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype

    fx = intrinsics[:, 0].view(batch, 1, 1)
    fy = intrinsics[:, 1].view(batch, 1, 1)
    cx = intrinsics[:, 2].view(batch, 1, 1)
    cy = intrinsics[:, 3].view(batch, 1, 1)

    y_px, x_px = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x_px.unsqueeze(0).expand(batch, -1, -1) - cx
    y = y_px.unsqueeze(0).expand(batch, -1, -1) - cy

    inv_depth = scale.view(batch, 1, 1) / (depth + 1e-6)
    inv_depth = torch.clamp(inv_depth, 0.0, 100.0)

    vx = v[:, 0].view(batch, 1, 1)
    vy = v[:, 1].view(batch, 1, 1)
    vz = v[:, 2].view(batch, 1, 1)

    flow_u = torch.clamp(inv_depth * (-fx * vx + x * vz), -100.0, 100.0)
    flow_v = torch.clamp(inv_depth * (-fy * vy + y * vz), -100.0, 100.0)
    return torch.stack([flow_u, flow_v], dim=1)

def compute_flow_target(flow_visual: torch.Tensor, flow_rot: torch.Tensor) -> torch.Tensor:

    return flow_visual - flow_rot

def compute_physics_loss_v16(
    flow_visual: torch.Tensor,
    flow_hat: torch.Tensor,
    bg_mask: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 0.001,
    scale_reg_weight: float = 0.01,
) -> torch.Tensor:

    flow_visual = torch.clamp(flow_visual, -100.0, 100.0)
    flow_hat = torch.clamp(flow_hat, -100.0, 100.0)

    residual = torch.sqrt(((flow_visual - flow_hat) ** 2).sum(dim=1) + 1e-8)
    residual = torch.clamp(residual, 0.0, 1000.0)
    charbonnier = torch.sqrt(residual**2 + eps**2) - eps

    mask = bg_mask.float()
    if mask.sum() > 0:
        loss = (charbonnier * mask).sum() / (mask.sum() + 1e-8)
    else:
        loss = charbonnier.mean()

    scale_clamped = torch.clamp(scale, 0.01, 100.0)
    scale_reg = scale_reg_weight * (torch.log(scale_clamped + 1e-6) ** 2).mean()
    return loss + scale_reg

class BPINNOutput(NamedTuple):
    logits: torch.Tensor
    probability: torch.Tensor
    uncertainty: torch.Tensor
    kl_divergence: torch.Tensor
    flow_hat: torch.Tensor
    v_pred: torch.Tensor
    scale: torch.Tensor
    flow_features: torch.Tensor
    depth_features: torch.Tensor
    imu_features: torch.Tensor

class BPINN(nn.Module):

    def __init__(self, config: Optional[BPINNConfig] = None) -> None:
        super().__init__()
        self.config = config or BPINNConfig()

        self.flow_encoder = FlowEncoder(hidden_dim=self.config.flow_hidden)
        self.depth_encoder_class = DepthEncoder(hidden_dim=self.config.depth_hidden)
        self.imu_encoder_class = IMUEncoder(
            input_channels=self.config.imu_channels,
            hidden_dim=self.config.imu_hidden,
            sequence_length=self.config.imu_sequence_length,
        )

        fusion_input_dim = (
            self.config.flow_hidden + self.config.depth_hidden + self.config.imu_hidden
        )
        self.fusion = BayesianFusionNetwork(
            input_dim=fusion_input_dim,
            hidden_dim=self.config.fusion_hidden,
            num_layers=self.config.fusion_layers,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
            use_flipout=self.config.use_flipout,
            prior_std=self.config.prior_std,
            posterior_std_init=self.config.posterior_std_init,
        )

        self.depth_encoder_phys = DepthEncoder(hidden_dim=self.config.depth_hidden)
        self.imu_encoder_phys = IMUEncoder(
            input_channels=self.config.imu_channels,
            hidden_dim=self.config.imu_hidden,
            sequence_length=self.config.imu_sequence_length,
        )
        self.translation_head = TranslationPredictor(
            imu_dim=self.config.imu_hidden,
            depth_dim=self.config.depth_hidden,
        )

    def forward(
        self,
        flow_visual: torch.Tensor,
        depth: torch.Tensor,
        imu: torch.Tensor,
        intrinsics: torch.Tensor,
        flow_rot: torch.Tensor,
        n_samples: int = 1,
    ) -> BPINNOutput:
        flow_features = self.flow_encoder(flow_visual)
        depth_features = self.depth_encoder_class(depth)
        imu_features = self.imu_encoder_class(imu)
        fused = torch.cat([flow_features, depth_features, imu_features], dim=-1)

        if n_samples > 1:
            logits_samples = []
            kl_sum = torch.tensor(0.0, device=flow_visual.device)
            for _ in range(n_samples):
                logits_i, kl_i = self.fusion(fused)
                logits_samples.append(logits_i)
                kl_sum = kl_sum + kl_i
            logits_stack = torch.stack(logits_samples, dim=0)
            logits = logits_stack.mean(dim=0)
            uncertainty = logits_stack.std(dim=0).mean(dim=-1)
            kl_divergence = kl_sum / n_samples
        else:
            logits, kl_divergence = self.fusion(fused)
            uncertainty = torch.zeros(logits.size(0), device=logits.device)

        probs = F.softmax(logits, dim=-1)
        authentic_prob = probs[:, 1] if probs.size(-1) > 1 else probs.squeeze(-1)

        depth_phys = depth if depth.dim() == 4 else depth.unsqueeze(1)
        depth_features_phys = self.depth_encoder_phys(depth_phys)
        imu_features_phys = self.imu_encoder_phys(imu)
        v_pred, scale = self.translation_head(imu_features_phys, depth_features_phys)

        depth_hw = depth.squeeze(1) if depth.dim() == 4 else depth
        flow_trans = compute_translational_flow_v16(v_pred, depth_hw, intrinsics, scale)
        flow_hat = flow_rot + flow_trans

        return BPINNOutput(
            logits=logits,
            probability=authentic_prob,
            uncertainty=uncertainty,
            kl_divergence=kl_divergence,
            flow_hat=flow_hat,
            v_pred=v_pred,
            scale=scale,
            flow_features=flow_features,
            depth_features=depth_features,
            imu_features=imu_features,
        )

    def set_classifier_generator(self, generator: Optional[torch.Generator]) -> None:
        for module in self.fusion.modules():
            if isinstance(module, FlipoutLinear):
                module.set_generator(generator)

    def predict_flow_only(
        self,
        depth: torch.Tensor,
        imu: torch.Tensor,
        intrinsics: torch.Tensor,
        flow_rot: torch.Tensor,
    ) -> torch.Tensor:
        depth_phys = depth if depth.dim() == 4 else depth.unsqueeze(1)
        depth_features = self.depth_encoder_phys(depth_phys)
        imu_features = self.imu_encoder_phys(imu)
        v_pred, scale = self.translation_head(imu_features, depth_features)
        depth_hw = depth.squeeze(1) if depth.dim() == 4 else depth
        return flow_rot + compute_translational_flow_v16(v_pred, depth_hw, intrinsics, scale)

    @torch.no_grad()
    def predict(
        self,
        flow_visual: torch.Tensor,
        depth: torch.Tensor,
        imu: torch.Tensor,
        intrinsics: torch.Tensor,
        flow_rot: torch.Tensor,
        threshold: float = 0.5,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        was_training = self.training
        self.eval()

        probs = []
        for _ in range(max(1, n_samples)):
            output = self(flow_visual, depth, imu, intrinsics, flow_rot)
            probs.append(output.probability)

        if was_training:
            self.train()

        prob_stack = torch.stack(probs, dim=0)
        mean_prob = prob_stack.mean(dim=0)
        uncertainty = prob_stack.std(dim=0)
        return mean_prob > threshold, mean_prob, uncertainty

class BPINNLoss(nn.Module):

    def __init__(
        self,
        kl_weight: float = 1e-4,
        physics_weight: float = 0.1,
        charbonnier_eps: float = 0.001,
        scale_reg_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.kl_weight = kl_weight
        self.physics_weight = physics_weight
        self.charbonnier_eps = charbonnier_eps
        self.scale_reg_weight = scale_reg_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        output: BPINNOutput,
        labels: torch.Tensor,
        flow_visual: torch.Tensor,
        bg_mask: torch.Tensor,
        num_batches: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_ce = self.ce_loss(output.logits, labels)
        loss_kl = output.kl_divergence / max(1, num_batches)
        loss_physics = compute_physics_loss_v16(
            flow_visual=flow_visual,
            flow_hat=output.flow_hat,
            bg_mask=bg_mask,
            scale=output.scale,
            eps=self.charbonnier_eps,
            scale_reg_weight=self.scale_reg_weight,
        )
        total_loss = loss_ce + self.kl_weight * loss_kl + self.physics_weight * loss_physics

        return total_loss, {
            "classification": float(loss_ce.detach().item()),
            "kl_divergence": float(loss_kl.detach().item()),
            "physics": float(loss_physics.detach().item()),
            "total": float(total_loss.detach().item()),
        }

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_bpinn(
    use_flipout: bool = True,
    flow_hidden: int = 128,
    depth_hidden: int = 64,
    imu_hidden: int = 64,
    fusion_hidden: int = 256,
    physics_weight: float = 0.1,
) -> Tuple[BPINN, BPINNLoss]:
    config = BPINNConfig(
        flow_hidden=flow_hidden,
        depth_hidden=depth_hidden,
        imu_hidden=imu_hidden,
        fusion_hidden=fusion_hidden,
        use_flipout=use_flipout,
    )
    return BPINN(config), BPINNLoss(physics_weight=physics_weight)
