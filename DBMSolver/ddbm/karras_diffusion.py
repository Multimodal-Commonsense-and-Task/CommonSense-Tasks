"""
Based on: https://github.com/crowsonkb/k-diffusion
"""

import numpy as np
import torch
from piq import LPIPS
from tqdm.auto import tqdm
import torch.distributed as dist


from .nn import mean_flat, append_dims, append_zero
from .random_util import BatchedSeedGenerator


class NoiseSchedule:
    def __init__(self):
        raise NotImplementedError

    def get_f_g2(self, t):
        raise NotImplementedError

    def get_alpha_rho(self, t):
        raise NotImplementedError

    def get_timestep_from_lambda(self, lambda_t):
        raise NotImplementedError
    
    def get_abc(self, t):
        alpha_t, alpha_bar_t, rho_t, rho_bar_t = self.get_alpha_rho(t)
        a_t, b_t, c_t = (
            (alpha_bar_t * rho_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t**2) / self.rho_T**2,
            (alpha_t * rho_bar_t * rho_t) / self.rho_T,
        )
        return a_t, b_t, c_t


class VPNoiseSchedule(NoiseSchedule):
    def __init__(self, beta_d=2, beta_min=0.1):
        self.beta_d, self.beta_min = beta_d, beta_min
        self.alpha_fn = lambda t: np.e ** (-0.5 * beta_min * t - 0.25 * beta_d * t**2)
        self.alpha_T = self.alpha_fn(1)
        self.rho_fn = lambda t: (np.e ** (beta_min * t + 0.5 * beta_d * t**2) - 1).sqrt()
        self.rho_T = self.rho_fn(torch.DoubleTensor([1])).item()
        
        self.lambda_inverse = lambda l: (-beta_min + ((beta_min**2) + (2 * beta_d * torch.log(1 + (-2*l).exp()))).sqrt()) / beta_d

        self.f_fn = lambda t: (-0.5 * beta_min - 0.5 * beta_d * t)
        self.g2_fn = lambda t: (beta_min + beta_d * t)

    def get_timestep_from_lambda(self, lambda_t):
        return self.lambda_inverse(lambda_t.to(torch.float64)).to(lambda_t.dtype)
    
    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class VENoiseSchedule(NoiseSchedule):
    def __init__(self, sigma_max=80.0):
        self.sigma_max = sigma_max
        self.alpha_fn = lambda t: torch.ones_like(t)
        self.alpha_T = 1
        self.rho_fn = lambda t: t
        self.rho_T = sigma_max

        self.lambda_inverse = lambda l: torch.exp(-l)
        
        self.f_fn = lambda t: torch.zeros_like(t)
        self.g2_fn = lambda t: 2 * t

    def get_timestep_from_lambda(self, lambda_t):
        return self.lambda_inverse(lambda_t.to(torch.float64)).to(lambda_t.dtype)
    
    def get_f_g2(self, t):
        t = t.to(torch.float64)
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = t.to(torch.float64)
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = (self.rho_T**2 - rho_t**2).sqrt()
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class I2SBNoiseSchedule(NoiseSchedule):
    def __init__(self, n_timestep=1000, beta_min=0.1, beta_max=1.0):
        self.n_timestep, self.linear_start, self.linear_end = (
            n_timestep,
            beta_min / n_timestep,
            beta_max / n_timestep,
        )
        betas = (
            torch.linspace(
                self.linear_start**0.5,
                self.linear_end**0.5,
                n_timestep,
                dtype=torch.float64,
            ).cuda()
            ** 2
        )
        betas = torch.cat(
            [
                betas[: self.n_timestep // 2],
                torch.flip(betas[: self.n_timestep // 2], dims=(0,)),
            ]
        )
        std_fwd = torch.sqrt(torch.cumsum(betas, dim=0))
        std_bwd = torch.sqrt(torch.flip(torch.cumsum(torch.flip(betas, dims=(0,)), dim=0), dims=(0,)))

        self.alpha_fn = lambda t: torch.ones_like(t).float()
        self.alpha_T = 1
        self.rho_fn = lambda t: std_fwd[t]
        self.rho_T = std_fwd[-1]
        self.rho_bar_fn = lambda t: std_bwd[t]

        self.f_fn = lambda t: torch.zeros_like(t).float()
        self.g2_fn = lambda t: betas[t]

    def get_f_g2(self, t):
        t = ((self.n_timestep - 1) * t).round().long()
        f, g2 = self.f_fn(t), self.g2_fn(t)
        return f, g2

    def get_alpha_rho(self, t):
        t = ((self.n_timestep - 1) * t).round().long()
        alpha_t = self.alpha_fn(t)
        alpha_bar_t = alpha_t / self.alpha_T
        rho_t = self.rho_fn(t)
        rho_bar_t = self.rho_bar_fn(t)
        return alpha_t, alpha_bar_t, rho_t, rho_bar_t


class PreCond:
    def __init__(self, ns):
        raise NotImplementedError

    def _get_scalings_and_weightings(self, t):
        raise NotImplementedError

    def get_scalings_and_weightings(self, t, ndim):
        c_skip, c_in, c_out, c_noise, weightings = self._get_scalings_and_weightings(t)
        c_skip, c_in, c_out, weightings = [append_dims(item, ndim) for item in [c_skip, c_in, c_out, weightings]]
        return c_skip, c_in, c_out, c_noise, weightings


class I2SBPreCond(PreCond):
    def __init__(self, ns, n_timestep=1000, t0=1e-4, T=1.0):
        self.ns = ns
        self.n_timestep = n_timestep
        self.noise_levels = torch.linspace(t0, T, n_timestep).cuda() * n_timestep

    def _get_scalings_and_weightings(self, t):
        _, _, rho_t, _ = self.ns.get_alpha_rho(t)
        c_skip = torch.ones_like(t)
        c_in = torch.ones_like(t)
        c_out = -rho_t
        c_noise = self.noise_levels[((self.n_timestep - 1) * t).round().long()]
        weightings = 1 / c_out**2
        return c_skip, c_in, c_out, c_noise, weightings


class DDBMPreCond(PreCond):
    def __init__(self, ns, sigma_data, cov_xy):
        self.ns, self.sigma_data, self.cov_xy = ns, sigma_data, cov_xy
        self.sigma_data_end = sigma_data

    def _get_scalings_and_weightings(self, t):
        a_t, b_t, c_t = self.ns.get_abc(t)
        A = a_t**2 * self.sigma_data_end**2 + b_t**2 * self.sigma_data**2 + 2 * a_t * b_t * self.cov_xy + c_t**2
        c_in = 1 / (A) ** 0.5
        c_skip = (b_t * self.sigma_data**2 + a_t * self.cov_xy) / A
        c_out = (
            a_t**2 * (self.sigma_data_end**2 * self.sigma_data**2 - self.cov_xy**2) + self.sigma_data**2 * c_t**2
        ) ** 0.5 * c_in
        c_noise = 1000 * 0.25 * torch.log(t + 1e-44)
        weightings = 1 / c_out**2
        return c_skip, c_in, c_out, c_noise, weightings


class KarrasDenoiser:
    def __init__(
        self,
        noise_schedule,
        precond,
        t_max=1.0,
        t_min=0.0001,
        loss_norm="lpips",
    ):

        self.t_max = t_max
        self.t_min = t_min

        self.noise_schedule = noise_schedule
        self.precond = precond

        self.loss_norm = loss_norm
        if loss_norm == "lpips":
            self.lpips_loss = LPIPS(replace_pooling=True, reduction="none")

    def bridge_sample(self, x0, xT, t, noise):
        a_t, b_t, c_t = [append_dims(item, x0.ndim) for item in self.noise_schedule.get_abc(t)]
        samples = a_t * xT + b_t * x0 + c_t * noise
        return samples

    def denoise(self, model, x_t, t, **model_kwargs):
        c_skip, c_in, c_out, c_noise, weightings = self.precond.get_scalings_and_weightings(t, x_t.ndim)
        model_output = model(c_in * x_t, c_noise, **model_kwargs)
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised, weightings

    def training_bridge_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        assert model_kwargs is not None
        xT = model_kwargs["xT"]
        mask = model_kwargs.pop("mask", None)
        if noise is None:
            noise = torch.randn_like(x_start)
        t = torch.minimum(t, torch.ones_like(t) * self.t_max)
        terms = {}

        x_t = self.bridge_sample(x_start, xT, t, noise)

        _, denoised, weights = self.denoise(model, x_t, t, **model_kwargs)

        if mask is not None:
            terms["xs_mse"] = mean_flat(mask * (denoised - x_start) ** 2)
            terms["mse"] = mean_flat(weights * mask * (denoised - x_start) ** 2)
        else:
            terms["xs_mse"] = mean_flat((denoised - x_start) ** 2)
            terms["mse"] = mean_flat(weights * (denoised - x_start) ** 2)

        terms["loss"] = terms["mse"]

        return terms


def karras_sample(
    diffusion,
    model,
    x_T,
    x_0,
    steps,
    dataset,
    use_2nd_order_sde: bool,
    mask=None,
    clip_denoised=True,
    model_kwargs=None,
    device=None,
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.0,
    eta=0.0,
    order=2,
    seed=None,
):
    assert sampler in [
        "heun",
        "ground_truth",
        "dbim",
        "dbim_high_order",
        "dbmsolver",
    ], "only these sampler is supported currently"

    if sampler == "ground_truth":
        gt = x_0.clamp(-1, 1)
        return (gt, [x_T], 0, [gt], [diffusion.t_max], None)
        
    if sampler == "heun":
        ts = get_sigmas_karras(steps, diffusion.t_min, diffusion.t_max - 1e-4, rho, device=device)
    elif sampler == 'dbmsolver':        
        ts = get_sigmas_karras(steps-1, diffusion.t_min, diffusion.t_max - 1e-4, device=device)
    else:
        ts = get_sigmas_uniform(steps, diffusion.t_min, diffusion.t_max - 1e-3, device=device)

    sample_fn = {
        "heun": sample_heun,
        "ground_truth": sample_ground_truth,
        "dbim": sample_dbim,
        "dbim_high_order": sample_dbim_high_order,
        "dbmsolver": sample_dbmsolver,
    }[sampler]

    sampler_args = dict(churn_step_ratio=churn_step_ratio, mask=mask, eta=eta, x_0=x_0, order=order, seed=seed, 
                        dataset=dataset, use_2nd_order_sde=use_2nd_order_sde)

    def denoiser(x_t, sigma):
        _, denoised, _ = diffusion.denoise(model, x_t, sigma, **model_kwargs)
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0, path, nfe, pred_x0, sigmas, noise = sample_fn(
        denoiser,
        diffusion,
        x_T,
        ts,
        **sampler_args,
    )
    if dist.get_rank() == 0:
        print("nfe:", nfe)

    return (
        x_0.clamp(-1, 1),
        [x.clamp(-1, 1) for x in path],
        nfe,
        [x.clamp(-1, 1) for x in pred_x0],
        sigmas,
        noise,
    )


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_uniform(n, t_min, t_max, device="cpu"):
    return torch.linspace(t_max, t_min, n + 1).to(device)


@torch.no_grad()
def sample_dbim_high_order(
    denoiser,
    diffusion,
    x,
    ts,
    mask=None,
    order=2,
    lower_order_final=True,
    seed=None,
    **kwargs,
):
    if order not in [2, 3]:
        raise NotImplementedError("Not supported")
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    u = diffusion.t_max
    if u == 1.0:
        u -= 5e-5
    u = [u for _ in range(order - 1)]
    xu_hat = [x0_hat.detach().clone() for _ in range(order - 1)]

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        # First Order Update, t < s
        if (lower_order_final and i + 1 == len(ts) - 1) or (i == 0):
            if dist.get_rank() == 0:
                print("Step order 1")
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]

            tmp_var = c_t / c_s
            coeff_xs = tmp_var
            coeff_x0_hat = b_t - tmp_var * b_s
            coeff_xT = a_t - tmp_var * a_s

            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            x_old = x
            x = coeff_xs * x_old + coeff_x0_hat * x0_hat + coeff_xT * x_T

        # Second Order Update, t < s < u
        elif order == 2 or i == 1:
            if dist.get_rank() == 0:
                print("Step order 2")
            a_u, b_u, c_u = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u, lambda_s, lambda_t = (
                torch.log(b_u / c_u),
                torch.log(b_s / c_s),
                torch.log(b_t / c_t),
            )

            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            h = lambda_t - lambda_s
            h2 = lambda_s - lambda_u
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat + (torch.exp(-h) + h - 1) * (x0_hat - xu_hat[-1]) / h2
            )
            x_old = x
            x = x_old * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        elif order == 3:
            if dist.get_rank() == 0:
                print("Step order 3")
            a_u1, b_u1, c_u1 = [
                append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-1] * ones)
            ]
            a_u2, b_u2, c_u2 = [
                append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(u[-2] * ones)
            ]
            a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u2, lambda_u1, lambda_s, lambda_t = (
                torch.log(b_u2 / c_u2),
                torch.log(b_u1 / c_u1),
                torch.log(b_s / c_s),
                torch.log(b_t / c_t),
            )
            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1

            h = lambda_t - lambda_s
            h1 = lambda_s - lambda_u1
            h2 = lambda_u1 - lambda_u2
            dx0_hat = ((x0_hat - xu_hat[-1]) * (2 * h1 + h2) / h1 - (xu_hat[-1] - xu_hat[-2]) * h1 / h2) / (h1 + h2)
            d2x0_hat = 2 * ((x0_hat - xu_hat[-1]) / h1 - (xu_hat[-1] - xu_hat[-2]) / h2) / (h1 + h2)
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat
                + (torch.exp(-h) + h - 1) * dx0_hat
                + (h**2 / 2 - h + 1 - torch.exp(-h)) * d2x0_hat
            )
            x_old = x
            x = x_old * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        u.append(s)
        u.pop(0)
        xu_hat.append(x0_hat)
        xu_hat.pop(0)

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())

    return x, path, nfe, pred_x0, ts, first_noise


@torch.no_grad()
def sample_dbim(
    denoiser,
    diffusion,
    x,
    ts,
    eta=1.0,
    mask=None,
    seed=None,
    **kwargs,
):
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        x0_hat = denoiser(x, s * ones)
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        a_s, b_s, c_s = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(s * ones)]
        a_t, b_t, c_t = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_abc(t * ones)]

        _, _, rho_s, _ = [append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(s * ones)]
        alpha_t, _, rho_t, _ = [
            append_dims(item, x0_hat.ndim) for item in diffusion.noise_schedule.get_alpha_rho(t * ones)
        ]

        omega_st = eta * (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()
        tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
        coeff_xs = tmp_var
        coeff_x0_hat = b_t - tmp_var * b_s
        coeff_xT = a_t - tmp_var * a_s

        noise = generator.randn_like(x0_hat)

        x = coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x + (1 if i != len(ts) - 2 else 0) * omega_st * noise

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise

@torch.no_grad()
def sample_dbmsolver(
    denoiser,
    diffusion,
    x,
    sigmas,
    dataset:str,
    use_2nd_order_sde:bool,
    mask=None,
    seed=None,
    **kwargs,
):
    """Implements the DBMSolver Algorithm"""
    x_T = x.clone()
    path = [x.detach().clone().cpu()]
    pred_x0 = []

    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))
    
    noise = torch.randn_like(x)
    
    if seed is None:
        noise = torch.randn_like(x)
    else:
        generator = BatchedSeedGenerator(seed)
        noise = generator.randn_like(x)
    square_m1 = lambda first, second: ((first ** 2) / (second ** 2)) - 1.

    nfe = 0
    
    sigma_max = diffusion.t_max
    
    alpha_T = diffusion.noise_schedule.alpha_T
    rho_T = diffusion.noise_schedule.rho_T
    
    sigma_T = torch.DoubleTensor([alpha_T * rho_T]).to(x.device)
    lambda_T = -torch.log(torch.DoubleTensor([rho_T])).to(x.device)
    
    denoised_zero = denoiser(x, sigma_max * s_in)
    nfe += 1
    if mask is not None:
        denoised_zero = denoised_zero * mask + x_T * (1 - mask)
    
    if use_2nd_order_sde:
        
        alpha, _, rho, __ = [ append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigmas[0] * s_in) ]
        sigma = (alpha * rho)
        
        temp_ratio = 0.5
        # sigma_temp = sigma_max + temp_ratio * (sigmas[0] - sigma_max)
        lambda_temp = torch.log(torch.DoubleTensor([1 / rho_T]).to(x.dtype).to(x.device)) + temp_ratio * (torch.log(1. / rho) - torch.log(torch.DoubleTensor([1 / rho_T]).to(x.dtype).to(x.device)))
        sigma_temp = diffusion.noise_schedule.get_timestep_from_lambda(lambda_temp).squeeze()
        
        alpha_temp, _, rho_temp, __ = [ append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigma_temp * s_in) ]
        sig_temp = (alpha_temp * rho_temp)
        t_temp = -torch.log(rho_temp)
        h_temp: torch.Tensor = t_temp - lambda_T
        
        x_temp = (sig_temp / sigma_T) * (-h_temp).exp() * x - alpha_temp * (-2 * h_temp).expm1() * denoised_zero
        x_temp = x_temp + sig_temp * (- (-2 * h_temp).expm1()).sqrt() * noise
        
        denoised_temp = denoiser(x_temp, sigma_temp * s_in)
        nfe += 1
        if mask is not None:
            denoised_temp = denoised_temp * mask + x_T * (1 - mask)
        
        alpha, _, rho, __ = [ append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigmas[0] * s_in) ]
        sigma = (alpha * rho)
        t_hat = -torch.log(rho)
        h: torch.Tensor = t_hat - lambda_T
        r = h_temp / h
        
        x_new = (sigma / sigma_T) * (-h).exp() * x - alpha * (-2 * h).expm1() * denoised_zero
        x_new += -0.5 * alpha * (-2 * h).expm1() * ((denoised_temp - denoised_zero) / r)
        x_new += sigma * (- (-2 * h).expm1()).sqrt() * noise
        x = x_new
    else:
        alpha, _, rho, __ = [ append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigmas[0] * s_in) ]
        sigma = (alpha * rho)
        t_hat = -torch.log(rho)
        h: torch.Tensor = t_hat - lambda_T
    
        x = (sigma / sigma_T) * (-h).exp() * x - alpha * (-2 * h).expm1() * denoised_zero
        x = x + sigma * (- (-2 * h).expm1()).sqrt() * noise

    path.append(x.detach().cpu())
    pred_x0.append(denoised_zero.detach().cpu())
    
    for j, i in enumerate(indices):
        
        sigma_hat = sigmas[i]
        sigma_next = sigmas[i + 1]
                
        if torch.all(sigma_next == 0):
            nfe += 1
            x, denoised = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                sigma_hat,
                sigma_next,
                stochastic=False,
            )
        else:
            denoised = denoiser(x, sigma_hat * s_in)
            
            if mask is not None:
                denoised = denoised * mask + x_T * (1 - mask)
            
            nfe += 2
            
            alpha_hat, _, rho_hat, _ = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigma_hat * s_in)]
            alpha_next, _, rho_next, _ = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigma_next * s_in)]
            
            sig_hat = alpha_hat * rho_hat
            sig_next = alpha_next * rho_next
            
            one_over_rho_hat = alpha_hat / sig_hat
            one_over_rho_next = alpha_next / sig_next
            one_over_rho_T = alpha_T / sigma_T

            ratio = 0.5
            if dataset == 'diode':
                lambda_mid = one_over_rho_hat.log() + ratio * (one_over_rho_next.log() - one_over_rho_hat.log())
                sigma_mid = diffusion.noise_schedule.get_timestep_from_lambda(lambda_mid).unique()
            elif dataset == 'e2h':
                sigma_mid = sigma_hat + ratio * (sigma_next - sigma_hat)
            alpha_mid, _, rho_mid, _ = [append_dims(item, x.ndim) for item in diffusion.noise_schedule.get_alpha_rho(sigma_mid * s_in)]
            sig_mid = alpha_mid * rho_mid
            one_over_rho_mid = alpha_mid / sig_mid
            
            common_factor = torch.sqrt(square_m1(one_over_rho_mid, one_over_rho_T) / square_m1(one_over_rho_hat, one_over_rho_T))
            
            first = x * alpha_mid / alpha_hat
            first = first * common_factor
            
            first = first * torch.square(one_over_rho_hat / one_over_rho_mid)
            
            second_a = denoised * alpha_mid
            second_b = 1. - torch.square(one_over_rho_T / one_over_rho_mid)
            second_c = 1. - (1. / common_factor)
            second = second_a * second_b * second_c
    
            third_a = x_T * alpha_mid / alpha_T
            third_b = torch.square(one_over_rho_T / one_over_rho_mid)
            third_c = 1. - common_factor
            third = third_a * third_b * third_c
            
            x_mid = first + second + third
            denoised_mid = denoiser(x_mid, sigma_mid * s_in)
            
            if mask is not None:
                denoised_mid = denoised_mid * mask + x_T * (1 - mask)
            
            common_factor = torch.sqrt(square_m1(one_over_rho_next, one_over_rho_T) / square_m1(one_over_rho_hat, one_over_rho_T))
            
            first = x * alpha_next / alpha_hat
            first = first * common_factor
            first = first * torch.square(one_over_rho_hat / one_over_rho_next)
            
            second_a = denoised * alpha_next
            
            second_b = 1. - torch.square(one_over_rho_T / one_over_rho_next)
            second_c = 1. - (1. / common_factor)
            second = second_a * second_b * second_c
    
            third_a = x_T * alpha_next / alpha_T
            third_b = torch.square(one_over_rho_T / one_over_rho_next)
            third_c = 1. - common_factor
            third = third_a * third_b * third_c
            
            _arctan_term_num_a = torch.sqrt(square_m1(one_over_rho_next, one_over_rho_T))
            _arctan_term_num_b = torch.sqrt(square_m1(one_over_rho_hat, one_over_rho_T))

            fourth_a = (denoised_mid - denoised) / (one_over_rho_mid.log() - one_over_rho_hat.log())

            fourth_b = alpha_next * square_m1(one_over_rho_next, one_over_rho_T) * torch.square(one_over_rho_T / one_over_rho_next)
            
            fourth_c_one = torch.arctan(_arctan_term_num_a) - torch.arctan(_arctan_term_num_b)
            fourth_c_one = fourth_c_one / _arctan_term_num_a
            fourth_c_two = (one_over_rho_next.log() - one_over_rho_hat.log() - 1.) + (1. / common_factor)
            
            fourth_c = fourth_c_one + fourth_c_two
            
            fourth = fourth_a * fourth_b * fourth_c
            
            x = first + second + third + fourth
            assert not x.isnan().any()
        
        path.append(x.detach().cpu())
        pred_x0.append(denoised.detach().cpu())
        
    return x, path, nfe, pred_x0, sigmas, None

@torch.no_grad()
def sample_ground_truth(
    denoiser,
    diffusion,
    x,
    ts,
    x0=None,
    **kwargs,
):
    assert x0 is not None
    x_T = x
    path = []
    pred_x0 = []

    ones = x.new_ones([x.shape[0]])
    indices = range(len(ts) - 1)
    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    noise = torch.randn_like(x0)
    first_noise = noise
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        x0_hat = denoiser(x, s * ones)
        noise = torch.randn_like(x0)
        x = diffusion.bridge_sample(x0, x_T, t * ones, noise)

        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())
        nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise


def get_d(denoiser, noise_schedule, x, x_T, t, stochastic):
    ones = x.new_ones([x.shape[0]])
    f_t, g2_t = [append_dims(item, x.ndim) for item in noise_schedule.get_f_g2(t * ones)]
    alpha_t, alpha_bar_t, _, rho_bar_t = [append_dims(item, x.ndim) for item in noise_schedule.get_alpha_rho(t * ones)]
    a_t, b_t, c_t = [append_dims(item, x.ndim) for item in noise_schedule.get_abc(t * ones)]
    denoised = denoiser(x, t * ones)
    grad_logq = -(x - (a_t * x_T + b_t * denoised)) / c_t**2
    grad_logpxTlxt = -(x - alpha_bar_t * x_T) / (alpha_t**2 * rho_bar_t**2)
    d = f_t * x - g2_t * ((0.5 if not stochastic else 1) * grad_logq - grad_logpxTlxt)
    return d, g2_t, denoised


def ddbm_simulate(denoiser, noise_schedule, x, x_T, t_cur, t_next, stochastic, second_order=False):
    dt = t_next - t_cur
    if isinstance(noise_schedule, I2SBNoiseSchedule):
        dt = dt * (noise_schedule.n_timestep - 1)
    d, g2_t, pred_x0 = get_d(denoiser, noise_schedule, x, x_T, t_cur, stochastic)
    x_new = x + d * dt + (0 if not stochastic else 1) * torch.randn_like(x) * ((dt).abs() ** 0.5) * g2_t.sqrt()
    if second_order:
        d_2, _, pred_x0 = get_d(denoiser, noise_schedule, x_new, x_T, t_next, stochastic)
        d_prime = (d + d_2) / 2
        x_new = (
            x + d_prime * dt + (0 if not stochastic else 1) * torch.randn_like(x) * ((dt).abs() ** 0.5) * g2_t.sqrt()
        )
    return x_new, pred_x0


@torch.no_grad()
def sample_heun(
    denoiser,
    diffusion,
    x,
    ts,
    churn_step_ratio=0.0,
    **kwargs,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    x_T = x
    path = []
    pred_x0 = []

    indices = range(len(ts) - 1)

    indices = tqdm(indices, disable=(dist.get_rank() != 0))

    nfe = 0
    assert churn_step_ratio < 1

    for _, i in enumerate(indices):

        if churn_step_ratio > 0:
            # 1 step euler
            t_hat = (ts[i + 1] - ts[i]) * churn_step_ratio + ts[i]
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                ts[i],
                t_hat,
                stochastic=True,
            )
            nfe += 1
            path.append(x.detach().cpu())
            pred_x0.append(_pred_x0.detach().cpu())
        else:
            t_hat = ts[i]

        # heun step
        if ts[i + 1] == 0:
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                t_hat,
                ts[i + 1],
                stochastic=False,
            )
            nfe += 1
        else:
            # Heun's method
            x, _pred_x0 = ddbm_simulate(
                denoiser,
                diffusion.noise_schedule,
                x,
                x_T,
                t_hat,
                ts[i + 1],
                stochastic=False,
                second_order=True,
            )
            nfe += 2

        path.append(x.detach().cpu())
        pred_x0.append(_pred_x0.detach().cpu())

    return x, path, nfe, pred_x0, ts, None
