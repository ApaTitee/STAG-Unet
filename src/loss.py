from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
	def __init__(self, smooth: float = 1e-6, ignore_index: int | None = 0) -> None:
		super().__init__()
		self.smooth = smooth
		self.ignore_index = ignore_index

	def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		num_classes = logits.shape[1]
		probabilities = F.softmax(logits, dim=1)
		target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

		losses = []
		for class_index in range(num_classes):
			if self.ignore_index is not None and class_index == self.ignore_index:
				continue
			pred_flat = probabilities[:, class_index].reshape(logits.shape[0], -1)
			target_flat = target_one_hot[:, class_index].reshape(logits.shape[0], -1)
			intersection = (pred_flat * target_flat).sum(dim=1)
			denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
			dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
			losses.append(1.0 - dice)

		if not losses:
			return logits.new_tensor(0.0)

		return torch.stack(losses, dim=0).mean()


class FocalLoss(nn.Module):
	def __init__(self, gamma: float = 2.0, alpha: float = 0.25, ignore_index: int | None = 0) -> None:
		super().__init__()
		self.gamma = gamma
		self.alpha = alpha
		self.ignore_index = ignore_index

	def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		log_probabilities = F.log_softmax(logits, dim=1)
		probabilities = log_probabilities.exp()
		batch_size, channels = logits.shape[:2]
		target = target.long().view(batch_size, -1)
		log_probabilities = log_probabilities.permute(0, 2, 3, 1).reshape(-1, channels)
		probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, channels)
		target = target.reshape(-1)

		if self.ignore_index is not None:
			valid_mask = target != self.ignore_index
			if not torch.any(valid_mask):
				return logits.new_tensor(0.0)
			target = target[valid_mask]
			log_probabilities = log_probabilities[valid_mask]
			probabilities = probabilities[valid_mask]

		target_log_prob = log_probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
		target_prob = probabilities.gather(1, target.unsqueeze(1)).squeeze(1)
		focal = -self.alpha * (1.0 - target_prob).pow(self.gamma) * target_log_prob
		return focal.mean()


class FocalDiceLoss(nn.Module):
	def __init__(
		self,
		alpha: float = 0.25,
		gamma: float = 2.0,
		focal_weight: float = 0.5,
		dice_weight: float = 0.5,
		ignore_index: int | None = 0,
	) -> None:
		super().__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.focal_weight = focal_weight
		self.dice_weight = dice_weight
		self.ignore_index = ignore_index
		self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, ignore_index=ignore_index)
		self.dice_loss = DiceLoss(ignore_index=ignore_index)

	def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
		focal = self.focal_loss(logits, target)
		dice = self.dice_loss(logits, target)
		return self.focal_weight * focal + self.dice_weight * dice

