from __future__ import annotations

import argparse
import os
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
	build_patch_records_from_manifest,
	build_sampler,
	build_train_transform,
	split_records_by_fold,
)
from src.loss import FocalDiceLoss  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import PatchConfig, ensure_dir, dice_score, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train STAG-Unet on BEETLE patches.")
	parser.add_argument("--csv-path", type=str, default=None)
	parser.add_argument("--data-root", type=str, default=None)
	parser.add_argument("--output-dir", type=str, default=None)
	parser.add_argument("--cache-root", type=str, default=None)
	parser.add_argument("--use-cache-manifest", action="store_true")
	parser.add_argument("--cache-only", action="store_true")
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
	parser.add_argument("--patch-sampling-rate", type=float, default=0.05)
	parser.add_argument("--use-stratified-sampling", action="store_true")
	parser.add_argument("--max-patch-candidates-per-wsi", type=int, default=256)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--max-wsis", type=int, default=None)
	parser.add_argument("--skip-blank", action="store_true")
	parser.add_argument("--pretrained", action="store_true")
	parser.add_argument("--no-pretrained", action="store_true")
	parser.add_argument("--early-stopping-patience", type=int, default=None)
	return parser.parse_args()


def _resolve_path(cli_value: str | None, env_var: str, fallback: str) -> Path:
	value = cli_value or os.getenv(env_var) or fallback
	return Path(value).expanduser().resolve()


def _prepare_runtime_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
	csv_path = _resolve_path(args.csv_path, "STAG_CSV_PATH", "data/data_overview.csv")
	data_root = _resolve_path(args.data_root, "STAG_DATA_ROOT", "data")
	output_dir = _resolve_path(args.output_dir, "STAG_OUTPUT_DIR", "outputs")

	if not csv_path.exists():
		raise FileNotFoundError(
			f"CSV not found: {csv_path}. Use --csv-path or set STAG_CSV_PATH to fix this."
		)
	if not data_root.exists() or not data_root.is_dir():
		raise FileNotFoundError(
			f"Data root not found: {data_root}. Use --data-root or set STAG_DATA_ROOT to fix this."
		)

	return csv_path, data_root, output_dir


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
	train_dataset = WSIPatchDataset(
		train_records,
		patch_size=patch_config.patch_size,
		transform=build_train_transform(),
		cache_root=args.cache_root,
		cache_only=args.cache_only,
	)
	val_dataset = WSIPatchDataset(
		val_records,
		patch_size=patch_config.patch_size,
		transform=build_eval_transform(),
		cache_root=args.cache_root,
		cache_only=args.cache_only,
	)

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
	epochs_without_improvement = 0
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
			epochs_without_improvement = 0
			torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, output_dir / "best.pt")
		else:
			epochs_without_improvement += 1

		print(
			f"[epoch {epoch:03d}/{args.epochs:03d}] "
			f"train_loss={train_loss:.6f} train_dice={train_dice:.6f} "
			f"val_loss={val_loss:.6f} val_dice={val_dice:.6f} best_val_loss={best_val_loss:.6f}",
			flush=True,
		)

		history = {
			"epoch": epoch,
			"train_loss": train_loss,
			"train_dice": train_dice,
			"val_loss": val_loss,
			"val_dice": val_dice,
			"best_val_loss": best_val_loss,
		}

		if args.early_stopping_patience is not None and args.early_stopping_patience > 0:
			if epochs_without_improvement >= args.early_stopping_patience:
				print(
					f"[early_stopping] stop at epoch {epoch}: no val_loss improvement for "
					f"{epochs_without_improvement} epochs (patience={args.early_stopping_patience})",
					flush=True,
				)
				break

	writer.close()
	return history


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	csv_path, data_root, output_dir = _prepare_runtime_paths(args)

	is_slurm_job = bool(os.getenv("SLURM_JOB_ID"))
	if not is_slurm_job:
		raise RuntimeError(
			"Training must be launched from a SLURM job. Use sbatch scripts/train.slurm or sbatch scripts/train_5fold_cached.slurm."
		)

	args.csv_path = str(csv_path)
	args.data_root = str(data_root)
	args.output_dir = str(output_dir)

	print(f"[runtime] csv_path={args.csv_path}")
	print(f"[runtime] data_root={args.data_root}")
	print(f"[runtime] output_dir={args.output_dir}")

	if args.use_cache_manifest:
		if args.cache_root is None:
			raise ValueError("--use-cache-manifest requires --cache-root.")
		records = build_patch_records_from_manifest(
			cache_root=args.cache_root,
			csv_path=args.csv_path,
			data_root=args.data_root,
			split="development",
			num_folds=args.num_folds,
		)
		print(f"[runtime] records_source=manifest records={len(records)}")
	else:
		patch_config = PatchConfig()
		records = build_patch_records(
			csv_path=args.csv_path,
			data_root=args.data_root,
			patch_config=patch_config,
			split="development",
			num_folds=args.num_folds,
			skip_blank=args.skip_blank,
			max_wsis=args.max_wsis,
			patch_sampling_rate=args.patch_sampling_rate,
			use_stratified_sampling=args.use_stratified_sampling,
			max_patch_candidates_per_wsi=args.max_patch_candidates_per_wsi,
			random_seed=args.seed,
		)
		print(f"[runtime] records_source=build_patch_records records={len(records)}")

	if not records:
		raise RuntimeError(
			"No patch records were generated. This usually means required WSI/mask files are missing. "
			f"Verify {args.data_root}/images/development/wsis and {args.data_root}/annotations/masks are populated, then re-run training."
		)

	if args.fold < 0 or args.fold >= args.num_folds:
		raise ValueError(f"fold must be within [0, {args.num_folds - 1}]")

	history = train_fold(args, records, args.fold)
	print(history)


if __name__ == "__main__":
	main()

