import pickle
import numpy as np
import time

def get_class_prompt_dict(classes, prompts):
    results_dict = {}
    for i, prompt in enumerate(prompts):
        results_dict[i] = {}
        for cls in classes:
            cls_original = cls
            # Add prefix (a / an) to cls
            if cls[0].lower() in ['a', 'e', 'i', 'o', 'u']:
                cls = "an " + cls
            else:
                cls = "a " + cls
            cls.replace("_", " ")
            if prompt.index('{class_name}') == 0:
                cls = cls[0].upper() + cls[1:]
            results_dict[i][cls_original] = prompt.format(class_name=cls).replace("_", " ")

    return results_dict
