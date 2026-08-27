import argparse

from .karras_diffusion import (
    KarrasDenoiser,
    VPNoiseSchedule,
    VENoiseSchedule,
    I2SBNoiseSchedule,
    DDBMPreCond,
    I2SBPreCond,
)
from .unet import UNetModel

NUM_CLASSES = 1000


def get_workdir(exp):
    workdir = f"./workdir/{exp}"
    return workdir


def sample_defaults():
    return dict(
        generator="determ",
        clip_denoised=True,
        sampler="euler",
        s_churn=0.0,
        s_tmin=0.002,
        s_tmax=80,
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        sigma_data=0.5,
        sigma_min=0.002,
        sigma_max=80.0,
        beta_d=2,
        beta_min=0.1,
        beta_max=1.0,
        cov_xy=0.0,
        image_size=64,
        in_channels=3,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        unet_type="adm",
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_new_attention_order=False,
        condition_mode=None,
        noise_schedule="ve",
    )
    return res


def create_model_and_diffusion(
    image_size,
    in_channels,
    class_cond,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    condition_mode,
    noise_schedule,
    sigma_data=0.5,
    sigma_min=0.002,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    beta_max=1.0,
    cov_xy=0.0,
    unet_type="adm",
):
    model = create_model(
        image_size,
        in_channels,
        num_channels,
        num_res_blocks,
        unet_type=unet_type,
        channel_mult=channel_mult,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        condition_mode=condition_mode,
    )
    if noise_schedule.startswith("vp"):
        ns = VPNoiseSchedule(beta_d=beta_d, beta_min=beta_min)
        precond = DDBMPreCond(ns, sigma_data=sigma_data, cov_xy=cov_xy)
    elif noise_schedule == "ve":
        ns = VENoiseSchedule(sigma_max=sigma_max)
        precond = DDBMPreCond(ns, sigma_data=sigma_data, cov_xy=cov_xy)
    elif noise_schedule.startswith("i2sb"):
        ns = I2SBNoiseSchedule(beta_max=beta_max, beta_min=beta_min)
        precond = I2SBPreCond(ns)

    diffusion = KarrasDenoiser(
        noise_schedule=ns,
        precond=precond,
        t_max=sigma_max,
        t_min=sigma_min,
    )
    return model, diffusion


def create_model(
    image_size,
    in_channels,
    num_channels,
    num_res_blocks,
    unet_type="adm",
    channel_mult="",
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    condition_mode=None,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if unet_type == "adm":
        return UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            # use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            condition_mode=condition_mode,
        )
    else:
        raise ValueError(f"Unsupported unet type: {unet_type}")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
