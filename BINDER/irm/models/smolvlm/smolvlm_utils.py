import numpy as np
from typing import Optional, Union, Unpack
from transformers.models.smolvlm.processing_smolvlm import SmolVLMProcessor
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, AllKwargsForChatTemplate
from dataclasses import dataclass

@dataclass
class VideoMetadata:
    total_num_frames: int
    fps: float
    duration: float
    video_backend: str


class SmolVLMCustomProcessor(SmolVLMProcessor):
    def apply_chat_template(
        self,
        frames,
        video_fps,
        chat_template: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str:
        if chat_template is None:
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "No chat template is set for this processor. Please either set the `chat_template` attribute, "
                    "or provide a chat template as an argument. See "
                    "https://huggingface.co/docs/transformers/main/en/chat_templating for more information."
                )
        
        tokenizer_template_kwargs = {'tools': None, 'documents': None, 'add_generation_prompt': True, 'continue_final_message': False, 'return_assistant_tokens_mask': False}

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": "dummy"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        ]

        total_num_frames = len(frames) # 32
        duration = float(total_num_frames) / float(video_fps) # 4.0

        metadata = VideoMetadata(
            total_num_frames=int(total_num_frames), fps=float(video_fps), duration=float(duration), video_backend="pyav"
        )
        metadata.frames_indices = np.arange(0, total_num_frames, 1)

        batch_images = []
        batch_audios = []
        batch_videos = [[frames]]
        batch_video_metadata = [[metadata]]
        mm_load_kwargs = {'num_frames': None, 'video_load_backend': 'pyav', 'video_fps': None, 'sampling_rate': 16000, 'sample_indices_fn': None, 'load_audio_from_video': False}
        
        conversations = self._process_messages_for_chat_template(
            conversations,
            batch_images=batch_images,
            batch_videos=batch_videos,
            batch_video_metadata=batch_video_metadata,
            **mm_load_kwargs,
        )
        # prompt = ['<|im_start|>User: You are provided the following series of three frames from a 0:00:02 [H:MM:SS] video.\n\nFrame from 00:00:<image>\nFrame from 00:01:<image>\nFrame from 00:02:<image>\n\nDescribe this video in detail<end_of_utterance>\nAssistant:']
        prompt = self.tokenizer.apply_chat_template( # GPT2TokenizerFast
            conversations,
            chat_template=chat_template,
            tokenize=False,
            return_dict=False,
            **tokenizer_template_kwargs,
        )[0]
        
        single_prompt = prompt
        if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
            kwargs["add_special_tokens"] = False

        out = self(
            text=prompt,
            images=batch_images if batch_images else None,
            videos=batch_videos if batch_videos else None,
            audio=batch_audios if batch_audios else None,
            **kwargs,
        )

        return out



if __name__ == "__main__":
    processor = SmolVLMCustomProcessor()
    processor.apply_chat_template(
        np.random.randint(0, 255, (32, 480, 480, 3)),
        video_fps,
        conversations,
    )
