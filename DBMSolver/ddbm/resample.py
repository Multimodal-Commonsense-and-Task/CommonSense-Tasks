import torch


def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "real-uniform":
        return RealUniformSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class RealUniformSampler:
    def __init__(self, diffusion):
        self.t_max = diffusion.t_max
        self.t_min = diffusion.t_min

    def sample(self, batch_size, device):
        ts = torch.rand(batch_size).to(device) * (self.t_max - self.t_min) + self.t_min
        return ts, torch.ones_like(ts)
