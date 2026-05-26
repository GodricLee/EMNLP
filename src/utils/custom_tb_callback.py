import os
import math
from typing import Optional
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

class CustomExpTensorBoardCallback(TrainerCallback):
    def __init__(self):
        self.writer: Optional[SummaryWriter] = None

    def attach_trainer(self, trainer) -> None:
        # Backwards-compatible placeholder; no trainer reference required
        return

    def _unwrap_model(self, model):
        """Unwrap common wrappers (Opacus GradSampleModule, PEFT) to access base model."""
        if model is None:
            return None
        # Unwrap Opacus GradSampleModule
        if hasattr(model, '_module'):
            model = model._module
        # Unwrap PEFT
        if hasattr(model, 'get_base_model'):
            model = model.get_base_model()
        return model

    def _is_world_zero(self, args, trainer=None) -> bool:
        try:
            if trainer is not None and hasattr(trainer, 'is_world_process_zero'):
                return bool(trainer.is_world_process_zero())
        except Exception:
            pass
        try:
            return int(getattr(args, 'process_index', 0)) == 0
        except Exception:
            return True

    def _get_base(self, model):
        return self._unwrap_model(model)

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get('trainer', None)
        if self.writer is not None or not self._is_world_zero(args, trainer):
            return
        # Create a separate events file next to HF logging directory
        log_dir = getattr(args, 'logging_dir', None)
        if not log_dir:
            # Fallback: follow HF default structure
            log_dir = os.path.join(args.output_dir, 'runs')
        os.makedirs(log_dir, exist_ok=True)
        try:
            self.writer = SummaryWriter(log_dir=log_dir, filename_suffix='.exp')
        except Exception:
            # Fallback: without suffix
            self.writer = SummaryWriter(log_dir=log_dir)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        trainer = kwargs.get('trainer', None)
        if self.writer is None or not self._is_world_zero(args, trainer):
            return
        step = int(state.global_step)
        # Total loss (HF reported loss); log as 'loss/total' for comparison
        try:
            if isinstance(logs, dict) and 'loss' in logs:
                self.writer.add_scalar('loss/total', float(logs['loss']), step)
        except Exception:
            pass
        base = self._get_base(model)
        if base is None:
            return
        # Loss breakdown
        try:
            if getattr(base, '_last_main_loss', None) is not None:
                self.writer.add_scalar('loss/main', float(base._last_main_loss), step)
        except Exception:
            pass
        try:
            if getattr(base, '_last_aux_contrib', None) is not None:
                self.writer.add_scalar('loss/aux_contrib', float(base._last_aux_contrib), step)
        except Exception:
            pass
        try:
            if getattr(base, '_last_kl_contrib', None) is not None:
                self.writer.add_scalar('loss/kl_contrib', float(base._last_kl_contrib), step)
        except Exception:
            pass
        try:
            if getattr(base, '_last_neg_aux_contrib', None) is not None:
                self.writer.add_scalar('loss/neg_aux_contrib', float(base._last_neg_aux_contrib), step)
        except Exception:
            pass
        # AUX (raw value and weight)
        try:
            if getattr(base, '_last_aux_loss', None) is not None:
                self.writer.add_scalar('aux/loss', float(base._last_aux_loss), step)
            if getattr(base, '_last_aux_lambda', None) is not None:
                self.writer.add_scalar('aux/lambda', float(base._last_aux_lambda), step)
        except Exception:
            pass
        # KL (raw value and weight)
        try:
            if getattr(base, '_last_kl_loss', None) is not None:
                self.writer.add_scalar('kl/loss', float(base._last_kl_loss), step)
            if getattr(base, '_kl_weight', None) is not None:
                self.writer.add_scalar('kl/weight', float(base._kl_weight), step)
        except Exception:
            pass
        # NEG-AUX (raw value and weight)
        try:
            if getattr(base, '_last_neg_aux_loss', None) is not None:
                self.writer.add_scalar('neg_aux/loss', float(base._last_neg_aux_loss), step)
            if getattr(base, '_neg_aux_weight', None) is not None:
                self.writer.add_scalar('neg_aux/weight', float(base._neg_aux_weight), step)
        except Exception:
            pass
        # [NEW] AUX gradient statistics (e.g. grad_norm_aux / grad_norm_main)
        try:
            aux_logs = getattr(base, 'aux_logs', None)
            if isinstance(aux_logs, dict):
                for key, val in aux_logs.items():
                    if val is not None:
                        self.writer.add_scalar(key, float(val), step)
        except Exception:
            pass
        # Modulation statistics
        try:
            stat = getattr(base, '_last_modulation_stats', None)
            if isinstance(stat, dict):
                if 'mask_frac' in stat:
                    self.writer.add_scalar('mod/mask_frac', float(stat['mask_frac']), step)
                if 'mask_frac_nonpad' in stat:
                    self.writer.add_scalar('mod/mask_frac_nonpad', float(stat['mask_frac_nonpad']), step)
                if 'mean_abs_delta' in stat:
                    self.writer.add_scalar('mod/mean_abs_delta', float(stat['mean_abs_delta']), step)
                if 'masked_mean_abs_delta' in stat:
                    self.writer.add_scalar('mod/masked_mean_abs_delta', float(stat['masked_mean_abs_delta']), step)
                if 'max_abs_delta' in stat:
                    self.writer.add_scalar('mod/max_abs_delta', float(stat['max_abs_delta']), step)
        except Exception:
            pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        trainer = kwargs.get('trainer', None)
        if self.writer is None or not self._is_world_zero(args, trainer) or not metrics:
            return
        step = int(state.global_step)
        try:
            if 'eval_loss' in metrics:
                ev_loss = float(metrics['eval_loss'])
                self.writer.add_scalar('eval/loss', ev_loss, step)
                try:
                    self.writer.add_scalar('eval/ppl', math.exp(ev_loss), step)
                except Exception:
                    pass
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        try:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
        except Exception:
            pass
        self.writer = None
