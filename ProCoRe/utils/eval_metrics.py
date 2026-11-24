from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
# from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from utils.bleu.bleu import Bleu
import numpy as np
import json
import torch
import string
import copy as cp
import os
import re

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x21, y21, x22, y22 = bbox2

    inter_w = max(0, min(x2, x22) - max(x1, x21))
    inter_h = max(0, min(y2, y22) - max(y1, y21))
    intersection_area = inter_w * inter_h

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x22 - x21) * max(0, y22 - y21)
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def parse_bbox_from_text(text):
    """
    Robustly parse bbox from generated text.
    Expected normalized format: [x1, y1, x2, y2]
    Returns:
        bbox (list[float] | None)
    """
    if text is None:
        return None

    text = text.strip().lower()

    # common assistant prefix removal
    for prefix in ["\nassistant:", "assistant:", "\nassistant", "assistant\n"]:
        idx = text.rfind(prefix)
        if idx != -1:
            text = text[idx + len(prefix):].strip()

    # remove obvious wrappers
    text = text.replace("\n", " ").strip()

    # 1) Try bracketed expression first
    bracket_match = re.search(r'\[([^\[\]]+)\]', text)
    if bracket_match:
        candidate = bracket_match.group(1)
        nums = re.findall(r'-?\d*\.?\d+', candidate)
        if len(nums) >= 4:
            try:
                return [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])]
            except Exception:
                pass

    # 2) Try tuple / list literal in the whole text
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
            return [float(x) for x in parsed]
    except Exception:
        pass

    # 3) Fallback: find first 4 numbers in the text
    nums = re.findall(r'-?\d*\.?\d+', text)
    if len(nums) >= 4:
        try:
            return [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])]
        except Exception:
            pass

    return None


def parse_gold_bbox(gold):
    """
    gold can be string like "[0.1,0.2,0.5,0.6]" or list/tuple.
    """
    if isinstance(gold, (list, tuple)) and len(gold) == 4:
        return [float(x) for x in gold]

    if isinstance(gold, str):
        gold = gold.strip()
        try:
            parsed = ast.literal_eval(gold)
            if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
                return [float(x) for x in parsed]
        except Exception:
            pass

        nums = re.findall(r'-?\d*\.?\d+', gold)
        if len(nums) >= 4:
            return [float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3])]

    return None


def denormalize_bbox_if_needed(bbox, image_file=None, image_size=None):
    """
    If bbox looks normalized ([0,1] range), convert to pixel coords using max(H,W),
    following code1 behavior.
    If already pixel coords, return as-is.
    """
    if bbox is None:
        return None

    if max(bbox) > 1.5:
        return bbox

    if image_size is None:
        if image_file is None:
            return bbox
        try:
            with Image.open(image_file).convert("RGB") as img:
                w, h = img.size
            image_size = (w, h)
        except Exception:
            return bbox

    max_wh = max(image_size)
    return [x * max_wh for x in bbox]


def matching_token_num(pred, gold):
    unique_pred = set(pred)
    unique_gold = set(gold)
    
    matching_token = unique_pred.intersection(unique_gold)
    
    return len(matching_token)

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

class NLPEvaluator(object):
    def __init__(self, prediction, verbose=False):
        # if not prediction_filename:
        #     raise IOError('Please input a valid prediction file.')

        self.verbose = verbose
        self.prediction = prediction#self.import_prediction(prediction_filename)

        self.tokenizer = PTBTokenizer()
        self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
            ]
        # if self.verbose:
        #     self.bertscorer = (BertScore(), "BertScore")

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print("Loading submission...")
        submission = json.load(open(prediction_filename))
        results = {}
        for vid_id in submission:
            results[vid_id] = submission[vid_id]
        return results

    def evaluate(self):
        self.scores = {}
        scores = self.example_evaluate(self.prediction)
        for metric, score in scores.items():
            if metric not in self.scores:
                self.scores[metric] = []
            self.scores[metric].append(score)
        return self.scores
        

    def example_evaluate(self, prediction):
        unique_index = 0
        cur_res = {}
        cur_gts = {}

        for pred in prediction:
            cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
            cur_gts[unique_index] = [{'caption': remove_nonascii(pred['gt_sentence'])}]
            unique_index += 1 

        all_scores = {}
        tokenize_res = self.tokenizer.tokenize(cur_res)
        tokenize_gts = self.tokenizer.tokenize(cur_gts)
        output = {}
        for scorer, method in self.scorers:
            if self.verbose:
                print('computing %s score...'%(scorer.method()))

            kargs = {'gts':tokenize_gts, 'res':tokenize_res}
            score, scores = scorer.compute_score(**kargs)

            if type(method) == list: 
                for sc, scs, m in zip(score, scores, method):
                    output[m] = float(sc)
                    if self.verbose: 
                        print("Calculated %s: %0.5f"%(m, sc))
            else:
                output[method] = np.mean(list(scores))
                if self.verbose: 
                    print("Calculated %s: %0.3f" % (method, output[method]))

        # if self.verbose: 
        #     scorer, method = self.bertscorer
        #     kargs = {'gts':gts, 'res':res}
        #     score, scores = scorer.compute_score(**kargs)
        #     output[method] = score 
        
        return output 


def can_infer_option(answer, choices):
    # Choices is a dictionary?
    # if 'Failed to obtain answer via API' in answer:
    #     return False

    # reject_to_answer = [
    #     "Sorry, I can't help with images of people yet.",
    #     "I can't process this file.",
    #     "I'm sorry, but without the image provided",
    #     'Cannot determine the answer'
    # ]
    # for err in reject_to_answer:
    #     if err in answer:
    #         return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3:
                return False
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)