class ReasoningFactory:
    @staticmethod
    def create(model_name: str, lora_path: str = None):
        if model_name == "smolvlm":
            from models.smolvlm.smolvlm import SmolVLMRReasoner
            return SmolVLMRReasoner()
        elif model_name == "qwen":
            from models.qwen.qwen import QwenReasoner
            return QwenReasoner(lora_path)
        elif model_name == "internvl":
            from models.internvl.internvl import InternVLReasoner
            return InternVLReasoner(lora_path)
        else:
            raise ValueError(f"Model {model_name} not found")