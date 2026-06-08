"""BaseTrainer — infrastructure base class for all cryo-ET trainers.

Extends ``deepinv.Trainer`` with:
- Gradient accumulation (``_grad_accum_steps``)
- Per-epoch timing and GPU memory logging
- CSV metric logging (``_metrics_dir``)
- Periodic + best-model checkpointing (``_ckpt_dir``)

Subclasses override the two science hooks:
  ``forward_pass(x, y, physics, train)``   — returns (x_net, y_net)
  ``val_step(epoch, logs)``                — fires after each val pass (FSC, PSNR, …)
"""
from __future__ import annotations

import time
from pathlib import Path

import deepinv as dinv
import torch

from .utils import PerfProbe, append_metrics_row


class ExponentialLRWithFloor(torch.optim.lr_scheduler.ExponentialLR):
    """ExponentialLR that clamps the LR at ``lr_min``."""
    def __init__(self, optimizer, gamma, lr_min: float = 0.0, **kwargs):
        self.lr_min = lr_min
        super().__init__(optimizer, gamma, **kwargs)
    def get_lr(self):  # type: ignore[override]
        return [max(self.lr_min, lr) for lr in super().get_lr()]


class BaseTrainer(dinv.Trainer):
    """Infrastructure base class.  All science lives in subclass hooks.

    Attributes set after construction:
      ``_metrics_dir``       (Path | None)  — CSV output directory
      ``_images_dir``        (Path | None)  — PNG output directory
      ``_ckpt_dir``          (Path | None)  — checkpoint directory
      ``_grad_accum_steps``  (int, default 1)
      ``_scheduler``         (LR scheduler | None)
      ``ckp_interval``       (int, default 10)
      ``_main_metric``       (str | None)   — logs key used for best-model saving
      ``_main_metric_higher_is_better`` (bool) — True for PSNR, False for FSC resolution
    """

    _main_metric: str | None = None          # set in subclass, e.g. "PSNR"
    _main_metric_higher_is_better: bool = True

    # ------------------------------------------------------------------
    # Science hooks — override in subclasses
    # ------------------------------------------------------------------

    def forward_pass(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        physics,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute network outputs.

        Default: standard single-input ``x_net = model(y)``.

        :returns: ``(x_net, y_net)`` where ``y_net`` is ``None`` unless the
            subclass computes a second estimate (e.g. equivariant training).
        """
        x_net = self.model_inference(y=y, physics=physics, x=x, train=train)
        return x_net, None

    def val_step(self, epoch: int, logs: dict) -> None:
        """Called once per val pass (after all val batches).

        Override to add FSC resolution, PSNR vs GT, or any custom metric.
        Results should be written directly into ``logs``.
        """

    # ------------------------------------------------------------------
    # Core training loop (infrastructure)
    # ------------------------------------------------------------------

    def compute_loss(self, physics, x, y, train=True, epoch=None, step=False):  # type: ignore[override]
        """Gradient-accumulation-aware training step with timing."""
        # ── Reset per-epoch accumulators on first batch ───────────────────
        if train:
            if epoch != getattr(self, "_current_train_epoch", None):
                self._current_train_epoch = epoch
                self._train_epoch_start   = time.perf_counter()
                self._train_batch_count   = 0
                self._epoch_fwd_model_time = 0.0
                self._epoch_fwd_loss_time  = 0.0
                self._epoch_bwd_time       = 0.0
            self._train_batch_count = getattr(self, "_train_batch_count", 0) + 1
        else:
            self._val_batch_count = getattr(self, "_val_batch_count", 0) + 1
            if getattr(self, "_val_start", None) is None:
                self._val_start = time.perf_counter()

        accum = max(1, getattr(self, "_grad_accum_steps", 1))

        # ── Gradient accumulation bookkeeping ────────────────────────────
        self._accum_count = getattr(self, "_accum_count", 0)
        at_window_start = self._accum_count % accum == 0
        self._accum_count += 1
        at_window_end = self._accum_count % accum == 0

        logs: dict = {}
        loss_total: torch.Tensor = torch.tensor(0.0)

        if train and step and at_window_start:
            self.optimizer.zero_grad(set_to_none=True)

        with torch.enable_grad() if train else torch.no_grad():
            # Forward pass
            with PerfProbe() as p_model:
                x_net, y_net = self.forward_pass(x, y, physics, train=train)

            # Loss computation
            if train or self.compute_eval_losses:
                with PerfProbe() as p_loss:
                    loss_total = torch.tensor(0.0, device=x.device)
                    for k, loss_fn in enumerate(self.losses):
                        loss = loss_fn(
                            x=x, x_net=x_net, y=y, y_net=y_net,
                            physics=physics, model=self.model, epoch=epoch,
                        )
                        loss_total = loss_total + loss.mean()
                        meters = self.logs_losses_train[k] if train else self.logs_losses_eval[k]
                        meters.update(loss.detach().cpu().numpy())
                        if len(self.losses) > 1:
                            logs[loss_fn.__class__.__name__] = meters.avg
                    meters = self.logs_total_loss_train if train else self.logs_total_loss_eval
                    meters.update(loss_total.item())
                    logs["TotalLoss"] = meters.avg
            else:
                p_loss = PerfProbe()

        # Backward + optimizer step
        if train:
            with PerfProbe() as p_bwd:
                (loss_total / accum).backward()

            norm = self.check_clip_grad()
            if norm is not None:
                logs["gradient_norm"] = self.check_grad_val.avg

            if step and at_window_end:
                self.optimizer.step()

            self._epoch_fwd_model_time += p_model.elapsed_s
            self._epoch_fwd_loss_time  += p_loss.elapsed_s
            self._epoch_bwd_time       += p_bwd.elapsed_s

            # ── Per-step CSV logging (rank 0 only) ───────────────────────
            if getattr(self, "_is_rank0", self.verbose):
                metrics_dir = getattr(self, "_metrics_dir", None)
                if metrics_dir is not None:
                    self._global_step = getattr(self, "_global_step", 0) + 1
                    step_row = {
                        "epoch": epoch,
                        "step": self._global_step,
                        "batch": self._train_batch_count,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
                    }
                    append_metrics_row(Path(metrics_dir) / "train_steps.csv", step_row)

        return loss_total, x_net, logs

    # ------------------------------------------------------------------
    # Logging, checkpointing
    # ------------------------------------------------------------------

    def log_metrics_mlops(self, logs: dict, step: int, train: bool = True) -> None:  # type: ignore[override]
        if not train:
            self.val_step(step, logs)



        # ── Rank-0 only below ─────────────────────────────────────────────
        if not getattr(self, "_is_rank0", self.verbose):
            return

        metrics_dir = getattr(self, "_metrics_dir", None)
        if metrics_dir is not None:
            row = {
                "epoch": step,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
            }
            append_metrics_row(Path(metrics_dir) / ("train_epochs.csv" if train else "val_epochs.csv"), row)

        if train:
            n = max(1, getattr(self, "_train_batch_count", 1))
            t_total   = time.perf_counter() - getattr(self, "_train_epoch_start", time.perf_counter())
            t_model   = getattr(self, "_epoch_fwd_model_time", 0.0)
            t_loss    = getattr(self, "_epoch_fwd_loss_time",  0.0)
            t_bwd     = getattr(self, "_epoch_bwd_time",       0.0)
            t_compute = t_model + t_loss + t_bwd
            if getattr(self, "_log_timing", True):
                alloc_gb  = torch.cuda.max_memory_allocated() / 1024**3
                total_gb  = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
                print(
                    f"[train ep={step}]  "
                    f"total={t_total:.1f}s  "
                    f"avg_compute/img={t_compute/n:.2f}s "
                    f"(model={t_model/n:.2f}s  loss={t_loss/n:.2f}s  bwd={t_bwd/n:.2f}s)  "
                    f"avg_total/img={t_total/n:.2f}s  "
                    f"max_gpu={alloc_gb:.2f}/{total_gb:.1f} GB  "
                    f"n={n}",
                    flush=True,
                )
            torch.cuda.reset_peak_memory_stats()
            self._val_start = time.perf_counter()
            self._val_batch_count = 0
        else:
            n_val = max(1, getattr(self, "_val_batch_count", 1))
            t_val = time.perf_counter() - getattr(self, "_val_start", time.perf_counter())
            n_log = getattr(self, "_log_every_n_epochs", 1)
            if step % n_log == 0:
                print(f"[time] val   ep={step}  total={t_val:.1f}s  per_img={t_val/n_val:.2f}s  n={n_val}", flush=True)
            self._val_start = None

        if train:
            scheduler = getattr(self, "_scheduler", None)
            if scheduler is not None:
                scheduler.step()
        # ── Checkpointing (rank 0 only, already inside verbose guard) ────
        ckpt_dir = getattr(self, "_ckpt_dir", None)
        if not train and ckpt_dir is not None:
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            ep = step
            state = {
                "epoch": ep,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler": getattr(self, "_scheduler", None) and self._scheduler.state_dict(),
            }
            if ep % getattr(self, "ckp_interval", 10) == 0:
                torch.save(state, Path(ckpt_dir) / f"ckp_{ep:04d}.pth")
                print(f"[ckpt] saved ckp_{ep:04d}.pth", flush=True)
            metric_key = getattr(self, "_main_metric", None)
            if metric_key:
                val_score = logs.get(metric_key)
                if val_score is not None:
                    higher = getattr(self, "_main_metric_higher_is_better", True)
                    sentinel = float("-inf") if higher else float("inf")
                    best = getattr(self, "_best_val_score", sentinel)
                    is_better = val_score > best if higher else val_score < best
                    if is_better:
                        self._best_val_score = val_score
                        torch.save(state, Path(ckpt_dir) / "ckp_best.pth")
                        direction = "↑" if higher else "↓"
                        print(f"[ckpt] new best {metric_key}={val_score:.4f} {direction} → ckp_best.pth", flush=True)

    def plot(self, epoch, physics, x, y, x_net, train=True):  # type: ignore[override]
        """Suppress the default deepinv plot."""
