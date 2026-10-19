"""Guidance-conditioned parsing module Phi_G (paper Eq. 3): Z_t -> (X_t, m_t)."""

from dataclasses import dataclass
from typing import List

from irm_task.base import ExecutionMode
from utils import match_targets

MANIP_CUES = {
    "watch_grasping": {
        ExecutionMode.CONTINUE: [
            "the object is held between the gripper fingers",
            "the gripper is holding the object",
            "the object is not slipping or being dropped",
            "the grasp looks successful",
        ],
        ExecutionMode.REPLAN: [
            "the object is not held between the gripper fingers",
            "the gripper is empty",
            "the object is slipping or being dropped",
            "the gripper missed the object",
            "the grasp does not look successful",
            "the grasp failed",
        ],
    },
    "grasping_refinement": {
        ExecutionMode.CONTINUE: [
            "the gripper looks well-positioned to grasp the object",
            "the gripper is aligned with the object",
            "the gripper can reach the object",
        ],
        ExecutionMode.ADJUST: [
            "the gripper does not look well-positioned to grasp the object",
            "the gripper is misaligned with the object",
            "the gripper is not aligned with the object",
            "the gripper cannot reach the object",
        ],
    },
}


@dataclass
class ParsedReport:
    objects: List[str]    # X_t
    mode: ExecutionMode   # m_t
    raw: str              # Z_t


class GuidanceParser:
    def __init__(self, embedding_model):
        self.model = embedding_model

    def parse(self, report: str, report_mode: str, target_objects, replan_targets=()) -> ParsedReport:
        if report_mode == "explore_multiple_targets":
            objects = match_targets(self.model, report, target_objects)
            mode = ExecutionMode.REPLAN if set(objects) & set(replan_targets) else ExecutionMode.CONTINUE
            return ParsedReport(objects, mode, report)
        if report_mode in MANIP_CUES:
            return ParsedReport([], self._manip_mode(report, MANIP_CUES[report_mode]), report)
        raise ValueError(f"Phi_G: unknown report_mode {report_mode!r}")

    def _manip_mode(self, report: str, cues) -> ExecutionMode:
        if report.strip().upper().startswith("ERROR"):
            return next(m for m in cues if m != ExecutionMode.CONTINUE)
        z = self.model.encode(report, normalize_embeddings=True, show_progress_bar=False)
        score = {
            m: (self.model.encode(p, normalize_embeddings=True, show_progress_bar=False) @ z).max()
            for m, p in cues.items()
        }
        return max(score, key=score.get)
