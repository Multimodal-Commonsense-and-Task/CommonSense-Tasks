import torch
from .glide_text2im.download import load_checkpoint
from .glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
from PIL import Image

class Glide:
    def __init__(self):
        pass
    
    def load_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.options = model_and_diffusion_defaults()
        self.options['use_fp16'] = torch.cuda.is_available()
        self.options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
        self.model, self.diffusion = create_model_and_diffusion(**self.options)
        self.guidance_scale = 3.0
        self.upsample_temp = 0.997
        self.model.eval()
        if torch.cuda.is_available():
            self.model.convert_to_fp16()
        self.model.to(self.device)
        self.model.load_state_dict(load_checkpoint('base', self.device))

        self.options_up = model_and_diffusion_defaults_upsampler()
        self.options_up['use_fp16'] = torch.cuda.is_available()
        self.options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
        self.model_up, self.diffusion_up = create_model_and_diffusion(**self.options_up)

        self.model_up.eval()
        if torch.cuda.is_available():
            self.model_up.convert_to_fp16()
        self.model_up.to(self.device)
        self.model_up.load_state_dict(load_checkpoint('upsample', self.device))

    def text2image(self, prompt, batch_size):
        tokens = self.model.tokenizer.encode(prompt)
        tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
            tokens, self.options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask(
            [], self.options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=torch.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device=self.device
            ),
            mask=torch.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=torch.bool,
                device=self.device,
            ),
        )
        
        # Create a classifier-free guidance sampling function
        def model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = torch.cat([half, half], dim=0)
            model_out = self.model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            return torch.cat([eps, rest], dim=1)

        # Sample from the base model.
        self.model.del_cache()
        samples = self.diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, self.options["image_size"], self.options["image_size"]),
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        self.model.del_cache()

        ##############################
        # Upsample the 64x64 samples #
        ##############################

        tokens = self.model_up.tokenizer.encode(prompt)
        tokens, mask = self.model_up.tokenizer.padded_tokens_and_mask(
            tokens, self.options_up['text_ctx']
        )

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples + 1) * 127.5).round() / 127.5 - 1,

            # Text tokens
            tokens=torch.tensor(
                [tokens] * batch_size, device=self.device
            ),
            mask=torch.tensor(
                [mask] * batch_size,
                dtype=torch.bool,
                device=self.device,
            ),
        )

        # Sample from the base model.
        self.model_up.del_cache()
        up_shape = (batch_size, 3, self.options_up["image_size"], self.options_up["image_size"])
        up_samples = self.diffusion_up.ddim_sample_loop(
            self.model_up,
            up_shape,
            noise=torch.randn(up_shape, device=self.device) * self.upsample_temp,
            device=self.device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        self.model_up.del_cache()
        
        scaled = ((up_samples + 1)*127.5).round().clamp(0,255).to(torch.uint8).cpu() # B 3 H W
        reshaped = scaled.permute(0, 2,3, 1) # B H W 3
        return Image.fromarray(reshaped[0].numpy())
