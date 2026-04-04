from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class ConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int) -> None:
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.block(inputs)


class CBAM(nn.Module):
	def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
		super().__init__()
		hidden = max(1, channels // reduction)
		self.mlp = nn.Sequential(
			nn.Linear(channels, hidden, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(hidden, channels, bias=False),
		)
		self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		batch_size, channels, _, _ = inputs.shape
		avg_pool = F.adaptive_avg_pool2d(inputs, 1).view(batch_size, channels)
		max_pool = F.adaptive_max_pool2d(inputs, 1).view(batch_size, channels)
		channel_attention = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(batch_size, channels, 1, 1)
		inputs = inputs * channel_attention

		avg_spatial = torch.mean(inputs, dim=1, keepdim=True)
		max_spatial, _ = torch.max(inputs, dim=1, keepdim=True)
		spatial_attention = torch.sigmoid(self.spatial(torch.cat([avg_spatial, max_spatial], dim=1)))
		return inputs * spatial_attention


class DecoderBlock(nn.Module):
	def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
		super().__init__()
		self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
		self.attention = CBAM(skip_channels)
		self.conv = ConvBlock(out_channels + skip_channels, out_channels)

	def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
		inputs = self.upsample(inputs)
		if inputs.shape[-2:] != skip.shape[-2:]:
			inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
		skip = self.attention(skip)
		return self.conv(torch.cat([inputs, skip], dim=1))


class EfficientNetB0Encoder(nn.Module):
	def __init__(self, pretrained: bool = True) -> None:
		super().__init__()
		weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
		backbone = efficientnet_b0(weights=weights)
		self.features = backbone.features

	def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
		features = []
		for index, module in enumerate(self.features):
			inputs = module(inputs)
			if index in {1, 2, 3, 4, 6, 8}:
				features.append(inputs)
		return features


class STAGUNet(nn.Module):
	def __init__(self, num_classes: int = 5, pretrained: bool = True) -> None:
		super().__init__()
		self.encoder = EfficientNetB0Encoder(pretrained=pretrained)
		self.bottleneck = ConvBlock(1280, 512)

		# Decoder matches EfficientNet feature map channels
		# Features extracted at indices {1, 2, 3, 4, 6, 8} have channels {16, 24, 40, 80, 192, 1280}
		self.decoder4 = DecoderBlock(512, 192, 256)
		self.decoder3 = DecoderBlock(256, 80, 128)
		self.decoder2 = DecoderBlock(128, 40, 64)
		self.decoder1 = DecoderBlock(64, 24, 32)
		self.decoder0 = DecoderBlock(32, 16, 16)

		self.segmentation_head = nn.Conv2d(16, num_classes, kernel_size=1)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		input_size = inputs.shape[-2:]
		features = self.encoder(inputs)
		stem, level1, level2, level3, level4, head = features
		x = self.bottleneck(head)
		x = self.decoder4(x, level4)
		x = self.decoder3(x, level3)
		x = self.decoder2(x, level2)
		x = self.decoder1(x, level1)
		x = self.decoder0(x, stem)
		logits = self.segmentation_head(x)
		if logits.shape[-2:] != input_size:
			logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
		return logits


def build_model(num_classes: int = 5, pretrained: bool = True) -> STAGUNet:
	return STAGUNet(num_classes=num_classes, pretrained=pretrained)

