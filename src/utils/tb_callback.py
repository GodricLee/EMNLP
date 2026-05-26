import os
import math
from typing import Optional
from transformers import TrainerCallback
from torch.utils.tensorboard import SummaryWriter

class ExactLossBreakdownCallback(TrainerCallback):
    def __init__(self, warn_after: int = 5, atol: float = 1e-6):
        self.writer: Optional[SummaryWriter] = None
        self._mismatch_count = 0
        self._warn_after = int(max(1, warn_after))
        self._atol = float(atol)
        # New: store aggregated result for each optimizer step (most recent) for on_log
        self._last_step_agg = None

    # [NEW] Backward-compatible placeholder for train_modulated.py (no trainer reference required)
    def attach_trainer(self, trainer) -> None:
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

    def _is_world_zero(self, trainer=None, args=None) -> bool:
        try:
            if trainer is not None and hasattr(trainer, 'is_world_process_zero'):
                return bool(trainer.is_world_process_zero())
        except Exception:
            pass
        try:
            return int(getattr(args, 'process_index', 0)) == 0
        except Exception:
            return True

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get('trainer', None)
        if self.writer is not None or not self._is_world_zero(trainer, args):
            return
        log_dir = getattr(args, 'logging_dir', None)
        if not log_dir:
            log_dir = os.path.join(args.output_dir, 'runs')
        os.makedirs(log_dir, exist_ok=True)
        try:
            self.writer = SummaryWriter(log_dir=log_dir, filename_suffix='.breakdown')
        except Exception:
            self.writer = SummaryWriter(log_dir=log_dir)

    def on_step_end(self, args, state, control, **kwargs):
        # At each optimizer step end, read and clear the micro-step breakdown buffer from the model and aggregate
        trainer = kwargs.get('trainer', None)
        if not self._is_world_zero(trainer, args):
            return
        model = kwargs.get('model', None)
        if model is None:
            return
        base = self._unwrap_model(model)
        if not hasattr(base, 'pop_breakdown_buffer'):
            return
        buf = base.pop_breakdown_buffer()
        if not buf:
            return
        # Aggregate: sum raw/contrib and token counts across micro-steps
        keys_sum = ['main_raw','aux_raw','kl_raw','neg_raw','aux_contrib','kl_contrib','neg_contrib','total_raw','main_tokens','aux_tokens']
        agg = {k: 0.0 for k in keys_sum}
        n = 0
        for bd in buf:
            n += 1
            for k in keys_sum:
                try:
                    agg[k] += float(bd.get(k, 0.0))
                except Exception:
                    pass
        agg['micro_steps'] = int(n)
        # Save aggregation for use in on_log
        self._last_step_agg = agg

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        trainer = kwargs.get('trainer', None)
        if self.writer is None or not self._is_world_zero(trainer, args):
            return
        step = int(state.global_step)
            # If per-step aggregation exists, use it instead of per-micro-step values to align with HF train/loss
        GA = 1
        try:
            GA = max(1, int(getattr(args, 'gradient_accumulation_steps', 1)))
        except Exception:
            GA = 1
        agg = self._last_step_agg
        if isinstance(agg, dict) and agg.get('micro_steps', 0) > 0:
            # `total_raw` is defined as the sum of `total_raw` over all micro-steps
            # in this optimizer step (the value actually used for backprop,
            # which HF internally divides by gradient accumulation `GA`).
            total_raw = float(agg.get('total_raw', 0.0))
            total_scaled = total_raw / float(GA)
            # For other components, use the sum (sum of raw/contrib values)
            main_raw = float(agg.get('main_raw', 0.0))
            aux_raw = float(agg.get('aux_raw', 0.0))
            kl_raw = float(agg.get('kl_raw', 0.0))
            neg_raw = float(agg.get('neg_raw', 0.0))
            aux_contrib = float(agg.get('aux_contrib', 0.0))
            kl_contrib = float(agg.get('kl_contrib', 0.0))
            neg_contrib = float(agg.get('neg_contrib', 0.0))
            self.writer.add_scalar('loss/main_raw', main_raw, step)
            self.writer.add_scalar('loss/aux_raw', aux_raw, step)
            self.writer.add_scalar('loss/kl_raw', kl_raw, step)
            self.writer.add_scalar('loss/neg_raw', neg_raw, step)
            self.writer.add_scalar('loss/aux_contrib', aux_contrib, step)
            self.writer.add_scalar('loss/kl_contrib', kl_contrib, step)
            self.writer.add_scalar('loss/neg_contrib', neg_contrib, step)
            self.writer.add_scalar('loss/total_raw', total_raw, step)
            self.writer.add_scalar('loss/total_scaled', total_scaled, step)
            # Tokens and flags (step totals)
            try:
                self.writer.add_scalar('tokens/main', int(agg.get('main_tokens', 0.0)), step)
            except Exception:
                pass
            try:
                self.writer.add_scalar('tokens/aux', int(agg.get('aux_tokens', 0.0)), step)
            except Exception:
                pass
        else:
            # Fallback: read the model's last breakdown directly
            bd = getattr(self._unwrap_model(model), '_last_breakdown', None) if model is not None else None
            if not isinstance(bd, dict):
                return
            main_raw = float(bd.get('main_raw', 0.0))
            aux_raw = float(bd.get('aux_raw', 0.0))
            kl_raw = float(bd.get('kl_raw', 0.0))
            neg_raw = float(bd.get('neg_raw', 0.0))
            aux_contrib = float(bd.get('aux_contrib', 0.0))
            kl_contrib = float(bd.get('kl_contrib', 0.0))
            neg_contrib = float(bd.get('neg_contrib', 0.0))
            total_raw = float(bd.get('total_raw', main_raw + aux_contrib + kl_contrib + neg_contrib))
            total_scaled = total_raw / float(GA)
            self.writer.add_scalar('loss/main_raw', main_raw, step)
            self.writer.add_scalar('loss/aux_raw', aux_raw, step)
            self.writer.add_scalar('loss/kl_raw', kl_raw, step)
            self.writer.add_scalar('loss/neg_raw', neg_raw, step)
            self.writer.add_scalar('loss/aux_contrib', aux_contrib, step)
            self.writer.add_scalar('loss/kl_contrib', kl_contrib, step)
            self.writer.add_scalar('loss/neg_contrib', neg_contrib, step)
            self.writer.add_scalar('loss/total_raw', total_raw, step)
            self.writer.add_scalar('loss/total_scaled', total_scaled, step)
        # HF reported loss
        hf_loss = None
        try:
            if isinstance(logs, dict) and 'loss' in logs:
                hf_loss = float(logs['loss'])
                self.writer.add_scalar('loss/hf_reported', hf_loss, step)
        except Exception:
            hf_loss = None
        if hf_loss is not None:
            diff = (total_scaled if isinstance(agg, dict) and agg.get('micro_steps', 0) > 0 else total_scaled) - hf_loss
            self.writer.add_scalar('loss/check_diff', diff, step)
            if abs(diff) > self._atol:
                self._mismatch_count += 1
                if self._mismatch_count >= self._warn_after:
                    try:
                        print(f"[LOSS-CHECK] step={step} diff={diff:.6e} total_scaled={total_scaled:.6f} hf_loss={hf_loss:.6f} (GA={GA})")
                    except Exception:
                        pass
        # Flags (use the model's last state)
        base = self._unwrap_model(model)
        bd_flag = getattr(base, '_last_breakdown', {}) if base is not None else {}
        try:
            self.writer.add_scalar('aux/enabled', 1 if bd_flag.get('aux_enabled', False) else 0, step)
            self.writer.add_scalar('kl/enabled', 1 if bd_flag.get('kl_enabled', False) else 0, step)
            self.writer.add_scalar('neg_aux/enabled', 1 if bd_flag.get('neg_enabled', False) else 0, step)
        except Exception:
            pass

        # ==== New: detailed AUX / tokens scalars ====
        # Ensure base model is current (may have been overridden above)
        base = self._unwrap_model(model)
        try:
            if base is not None:
                # Record fresh/replay supervised token counts separately
                fresh_tok = int(getattr(base, '_last_aux_tokens_fresh_supervised', 0) or 0)
                replay_tok = int(getattr(base, '_last_aux_tokens_replay', 0) or 0)
                self.writer.add_scalar('tokens/aux_fresh', fresh_tok, step)
                self.writer.add_scalar('tokens/aux_replay', replay_tok, step)
        except Exception:
            pass
        try:
            if base is not None:
                # Requires the model to provide: _last_aux_fresh_loss and _last_aux_replay_loss
                fresh_loss = getattr(base, '_last_aux_fresh_loss', None)
                replay_loss = getattr(base, '_last_aux_replay_loss', None)
                if fresh_loss is not None:
                    self.writer.add_scalar('aux/fresh_loss', float(fresh_loss), step)
                if replay_loss is not None:
                    self.writer.add_scalar('aux/replay_loss', float(replay_loss), step)
        except Exception:
            pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        trainer = kwargs.get('trainer', None)
        if self.writer is None or not self._is_world_zero(trainer, args) or not metrics:
            return
        step = int(state.global_step)
        if 'eval_loss' in metrics:
            ev_loss = float(metrics['eval_loss'])
            self.writer.add_scalar('eval/loss', ev_loss, step)
            try:
                self.writer.add_scalar('eval/ppl', math.exp(ev_loss), step)
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
