import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class BaseReasoner:
    def __init__(self):
        self.model = None
        self.processor = None
        self.load_model()

    def load_model(self):
        pass

    def preprocess(self, image_batch: list[np.ndarray]) -> list[np.ndarray]:
        pass
    
    def generate(self, image_batch: list[np.ndarray]) -> str:
        pass
    
    def run_from_dict(self, result_dict: dict) -> str:
        pass

    def run_from_mp4(self, video_path: str, prompt: str) -> str:
        pass

    def run_from_pickle(self, pickle_path: str, prompt: str) -> str:
        pass
    
    def save_video_summary(self, frames, prompt, result, sample_count=8, save_video=False, filename=None):
        """
        frames: np.ndarray of shape (T, H, W, 3)
        prompt, result: str
        sample_count: number of frames to sample
        """
        if not save_video:
            return
        total_frames = frames.shape[0]
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        sampled = frames[indices]  # (sample_count, H, W, 3)

        separator_width = 5
        H, W, _ = sampled[0].shape
        separator = np.zeros((H, separator_width, 3), dtype=np.uint8)

        frames_with_separators = []
        for i, frame in enumerate(sampled):
            frames_with_separators.append(frame)
            if i < len(sampled) - 1:
                frames_with_separators.append(separator)

        concatenated = np.concatenate(frames_with_separators, axis=1)  # (H, W * N + sep, 3)

        text = f"# Frames: {total_frames}\n\n\nPrompt: {prompt}\n\n\nResult:\n\n{result}"

        try:
            font = ImageFont.truetype("./DejaVuSans.ttf", size=64)
        except:
            font = ImageFont.load_default()
            print(f"[Warning] Font not found. Using default font.")

        max_width = concatenated.shape[1] - 20
        draw_dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        wrapped_lines = []
        for paragraph in text.split("\n"):
            if paragraph.strip() == "":
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(textwrap.wrap(paragraph, width=120))

        line_spacing = 10
        text_heights = [draw_dummy.textbbox((0, 0), line, font=font)[3] for line in wrapped_lines]
        text_img_height = sum(text_heights) + line_spacing * len(wrapped_lines) + 20

        text_img = Image.new("RGB", (concatenated.shape[1], text_img_height), (255, 255, 255))
        draw = ImageDraw.Draw(text_img)
        y = 10
        for line, h in zip(wrapped_lines, text_heights):
            draw.text((10, y), line, fill=(0, 0, 0), font=font)
            y += h + line_spacing

        combined = np.vstack((concatenated, np.array(text_img)))
        
        Image.fromarray(combined).save(f"{filename}.png")