from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

if __package__ is None or __package__ == "":
	sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.dataset import (  # noqa: E402
	WSIPatchDataset,
	build_dataloader,
	build_eval_transform,
	build_patch_records,
	build_sampler,
	build_train_transform,
	split_records_by_fold,
)
from src.loss import FocalDiceLoss  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import PatchConfig, ensure_dir, dice_score, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train STAG-Unet on BEETLE patches.")
	parser.add_argument("--csv-path", type=str, default="data/data_overview.csv")
	parser.add_argument("--data-root", type=str, default="data")
	parser.add_argument("--output-dir", type=str, default="outputs")
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--learning-rate", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--warmup-epochs", type=int, default=5)
	parser.add_argument("--fold", type=int, default=0)
	parser.add_argument("--num-folds", type=int, default=5)
	parser.add_argument("--num-workers", type=int, default=4)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num-classes", type=int, default=5)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--max-wsis", type=int, default=None)
	parser.add_argument("--skip-blank", action="store_true")
	parser.add_argument("--pretrained", action="store_true")
	parser.add_argument("--no-pretrained", action="store_true")
	return parser.parse_args()


def build_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int) -> SequentialLR:
	warmup_epochs = min(warmup_epochs, max(0, epochs - 1))
	main_epochs = max(1, epochs - warmup_epochs)
	warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup_epochs))
	cosine_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs)
	return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])


def run_epoch(model: nn.Module, dataloader, criterion, optimizer, device: torch.device, train: bool) -> tuple[float, float]:
	total_loss = 0.0
	total_dice = 0.0
	total_batches = 0

	if train:
		model.train()
	else:
		model.eval()

	iterator = tqdm(dataloader, desc="train" if train else "val", leave=False)
	for batch in iterator:
		images = batch["image"].to(device, non_blocking=True)
		masks = batch["mask"].to(device, non_blocking=True)

		with torch.set_grad_enabled(train):
			logits = model(images)
			if logits.shape[-2:] != masks.shape[-2:]:
				logits = torch.nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
			loss = criterion(logits, masks)

			if train:
				optimizer.zero_grad(set_to_none=True)
				loss.backward()
				optimizer.step()

		total_loss += float(loss.detach().cpu())
		total_dice += dice_score(logits.detach().cpu(), masks.detach().cpu(), num_classes=logits.shape[1])
		total_batches += 1

		iterator.set_postfix(loss=total_loss / max(1, total_batches), dice=total_dice / max(1, total_batches))

	return total_loss / max(1, total_batches), total_dice / max(1, total_batches)


def train_fold(args: argparse.Namespace, records, fold_index: int) -> dict[str, float]:
	output_dir = ensure_dir(Path(args.output_dir) / f"fold_{fold_index}")
	writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

	train_records, val_records = split_records_by_fold(records, fold_index)
	patch_config = PatchConfig()
	train_dataset = WSIPatchDataset(train_records, patch_size=patch_config.patch_size, transform=build_train_transform())
	val_dataset = WSIPatchDataset(val_records, patch_size=patch_config.patch_size, transform=build_eval_transform())

	train_loader = build_dataloader(
		train_dataset,
		batch_size=args.batch_size,
		sampler=build_sampler(train_records),
		num_workers=args.num_workers,
	)
	val_loader = build_dataloader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
	)

	device = torch.device(args.device)
	model = build_model(num_classes=args.num_classes, pretrained=args.pretrained and not args.no_pretrained).to(device)
	criterion = FocalDiceLoss(alpha=0.25, gamma=2.0, ignore_index=0)
	optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
	scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)

	best_val_loss = float("inf")
	history: dict[str, float] = {}

	for epoch in range(1, args.epochs + 1):
		train_loss, train_dice = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
		val_loss, val_dice = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
		scheduler.step()

		writer.add_scalar("loss/train", train_loss, epoch)
		writer.add_scalar("loss/val", val_loss, epoch)
		writer.add_scalar("dice/train", train_dice, epoch)
		writer.add_scalar("dice/val", val_dice, epoch)
		writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

		if epoch % 10 == 0:
			torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, output_dir / f"checkpoint_epoch_{epoch:03d}.pt")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, output_dir / "best.pt")

		history = {
			"train_loss": train_loss,
			"train_dice": train_dice,
			"val_loss": val_loss,
			"val_dice": val_dice,
			"best_val_loss": best_val_loss,
		}

	writer.close()
	return history


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	patch_config = PatchConfig()
	records = build_patch_records(
		csv_path=args.csv_path,
		data_root=args.data_root,
		patch_config=patch_config,
		split="development",
		num_folds=args.num_folds,
		skip_blank=args.skip_blank,
		max_wsis=args.max_wsis,
	)

	if not records:
		raise RuntimeError(
			"No patch records were generated. This usually means required WSI/mask files are missing. "
			"Verify data/images/development/wsis and data/annotations/masks are populated, then re-run training."
		)

	if args.fold < 0 or args.fold >= args.num_folds:
		raise ValueError(f"fold must be within [0, {args.num_folds - 1}]")

	history = train_fold(args, records, args.fold)
	print(history)


if __name__ == "__main__":
	main()

