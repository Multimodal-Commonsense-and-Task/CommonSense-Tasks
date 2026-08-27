import copy
import functools
import os

import numpy as np

import blobfile as bf
import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam

from . import dist_util, logger
from .nn import update_ema

from ddbm.random_util import get_generator

import glob

import wandb


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        train_data,
        test_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        test_interval,
        save_interval,
        save_interval_for_preemption,
        resume_checkpoint,
        workdir,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        total_training_steps=10000000,
        augment_pipe=None,
        train_mode="ddbm",
        resume_train_flag=False,
        **sample_kwargs,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = train_data
        self.test_data = test_data
        self.image_size = model.image_size
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        self.log_interval = log_interval
        self.workdir = workdir
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.save_interval_for_preemption = save_interval_for_preemption
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = total_training_steps

        self.train_mode = train_mode

        self.step = 0
        self.resume_train_flag = resume_train_flag
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)

        self._load_and_sync_parameters()
        if not self.resume_train_flag:
            self.resume_step = 0

        self.opt = RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
        else:
            self.ema_params = [copy.deepcopy(list(self.model.parameters())) for _ in range(len(self.ema_rate))]

        if torch.cuda.is_available():
            self.use_ddp = True
            local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA. " "Gradients will not be synchronized properly!")
            self.use_ddp = False
            self.ddp_model = self.model

        self.step = self.resume_step

        self.generator = get_generator(sample_kwargs["generator"], self.batch_size, 42)
        self.sample_kwargs = sample_kwargs

        self.augment = augment_pipe

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            if self.resume_train_flag:
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                logger.log("Resume step: ", self.resume_step)

            self.model.load_state_dict(torch.load(resume_checkpoint, map_location="cpu"))
            self.model.to(dist_util.dev())

            dist.barrier()

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(list(self.model.parameters()))

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = torch.load(ema_checkpoint, map_location=dist_util.dev())
            ema_params = [state_dict[name] for name, _ in self.model.named_parameters()]

            dist.barrier()
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if main_checkpoint.split("/")[-1].startswith("freq"):
            prefix = "freq_"
        else:
            prefix = ""
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"{prefix}opt_{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = torch.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)
            dist.barrier()

    def run_loop(self):
        while True:
            for batch, cond, _ in self.data:

                if "inpaint" in self.workdir:
                    _, mask, label = _
                else:
                    mask = None

                if not (not self.lr_anneal_steps or self.step < self.total_training_steps):
                    # Save the last checkpoint if it wasn't already saved.
                    if (self.step - 1) % self.save_interval != 0:
                        self.save()
                    return

                if self.augment is not None:
                    batch, _ = self.augment(batch)
                if isinstance(cond, torch.Tensor) and batch.ndim == cond.ndim:
                    cond = {"xT": cond}
                else:
                    cond["xT"] = cond["xT"]
                if mask is not None:
                    # cond["mask"] = mask
                    cond["y"] = label

                took_step = self.run_step(batch, cond)
                if took_step and self.step % self.log_interval == 0:
                    logs = logger.dumpkvs()

                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)

                if took_step and self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

                    test_batch, test_cond, _ = next(iter(self.test_data))
                    if "inpaint" in self.workdir:
                        _, mask, label = _
                    else:
                        mask = None
                    if isinstance(test_cond, torch.Tensor) and test_batch.ndim == test_cond.ndim:
                        test_cond = {"xT": test_cond}
                    else:
                        test_cond["xT"] = test_cond["xT"]
                    if mask is not None:
                        # test_cond["mask"] = mask
                        test_cond["y"] = label
                    self.run_test_step(test_batch, test_cond)
                    logs = logger.dumpkvs()

                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)

                if took_step and self.step % self.save_interval_for_preemption == 0:
                    self.save(for_preemption=True)

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        logger.logkv_mean("lg_loss_scale", np.log2(self.scaler.get_scale()))
        self.scaler.unscale_(self.opt)

        def _compute_norms():
            grad_norm = 0.0
            param_norm = 0.0
            for p in self.model.parameters():
                with torch.no_grad():
                    param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
                    if p.grad is not None:
                        grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
            return np.sqrt(grad_norm), np.sqrt(param_norm)

        grad_norm, param_norm = _compute_norms()

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        self.scaler.step(self.opt)
        self.scaler.update()
        self.step += 1
        self._update_ema()

        self._anneal_lr()
        self.log_step()
        return True

    def run_test_step(self, batch, cond):
        with torch.no_grad():
            self.forward_backward(batch, cond, train=False)

    def forward_backward(self, batch, cond, train=True):
        if train:
            self.opt.zero_grad()
        assert batch.shape[0] % self.microbatch == 0
        num_microbatches = batch.shape[0] // self.microbatch
        for i in range(0, batch.shape[0], self.microbatch):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                micro = batch[i : i + self.microbatch].to(dist_util.dev())
                micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev()) for k, v in cond.items()}
                last_batch = (i + self.microbatch) >= batch.shape[0]
                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

                if self.train_mode == "ddbm":
                    compute_losses = functools.partial(
                        self.diffusion.training_bridge_losses,
                        self.ddp_model,
                        micro,
                        t,
                        model_kwargs=micro_cond,
                    )
                else:
                    raise NotImplementedError()

                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

                loss = (losses["loss"] * weights).mean() / num_microbatches
            log_loss_dict(self.diffusion, t, {k if train else "test_" + k: v * weights for k, v in losses.items()})
            if train:
                self.scaler.scale(loss).backward()

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model.parameters(), rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self, for_preemption=False):
        def maybe_delete_earliest(filename):
            wc = filename.split(f"{(self.step):06d}")[0] + "*"
            freq_states = list(glob.glob(os.path.join(get_blob_logdir(), wc)))
            if len(freq_states) > 3000:
                earliest = min(freq_states, key=lambda x: x.split("_")[-1].split(".")[0])
                os.remove(earliest)

        # if dist.get_rank() == 0 and for_preemption:
        #     maybe_delete_earliest(get_blob_logdir())
        def save_checkpoint(rate, params):
            state_dict = self.model.state_dict()
            for i, (name, _) in enumerate(self.model.named_parameters()):
                assert name in state_dict
                state_dict[name] = params[i]
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model_{(self.step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step):06d}.pt"
                if for_preemption:
                    filename = f"freq_{filename}"
                    maybe_delete_earliest(filename)

                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    torch.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            filename = f"opt_{(self.step):06d}.pt"
            if for_preemption:
                filename = f"freq_{filename}"
                maybe_delete_earliest(filename)

            with bf.BlobFile(
                bf.join(get_blob_logdir(), filename),
                "wb",
            ) as f:
                torch.save(self.opt.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, list(self.model.parameters()))
        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/model_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    if main_checkpoint.split("/")[-1].startswith("freq"):
        prefix = "freq_"
    else:
        prefix = ""
    filename = f"{prefix}ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
