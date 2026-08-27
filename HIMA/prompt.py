from prompts.multi_lm_prompt import *


class Prompt:
    def __init__(self, race):
        self.race = race

    def generate_prompts(self):
        MULTI_LM_PROMPT = multi_lm_prompt(self.race)
        return [MULTI_LM_PROMPT['system'], MULTI_LM_PROMPT['input'], MULTI_LM_PROMPT['output']]

    def generate_vision_prompts(self, target):
        return f'This is StarCraft II real game image. Where can I build a {target}? Respond strictly with a single (x, y) coordinate — no explanations, no additional text, just the coordinate.'
