from sentence_transformers import SentenceTransformer

from guidance_parser import GuidanceParser
from irm_task.base import ExecutionMode as M

if __name__ == "__main__":
    phi = GuidanceParser(SentenceTransformer("sentence-transformers/all-mpnet-base-v2"))
    targets = ["black toy", "blue box", "red box", "apple"]

    p = phi.parse("a red box, a black plush toy, a chair", "explore_multiple_targets", targets, ["black toy"])
    assert sorted(p.objects) == ["black toy", "red box"] and p.mode == M.REPLAN, p
    p = phi.parse("a white table, a chair", "explore_multiple_targets", targets, ["black toy"])
    assert p.objects == [] and p.mode == M.CONTINUE, p

    for report, mode in [
        ("The cup is firmly held between the gripper fingers and the grasp looks successful.", M.CONTINUE),
        ("The grasp does not look successful, the bottle was dropped.", M.REPLAN),
        ("The gripper is empty and the cup remains on the table.", M.REPLAN),
        ("ERROR: VLM response timeout", M.REPLAN),
    ]:
        assert phi.parse(report, "watch_grasping", ["cup"]).mode == mode, report
    for report, mode in [
        ("The gripper is directly above the cup and looks well-positioned to grasp it.", M.CONTINUE),
        ("The gripper is offset to the left of the cup and seems misaligned.", M.ADJUST),
        ("ERROR: VLM response timeout", M.ADJUST),
    ]:
        assert phi.parse(report, "grasping_refinement", ["cup"]).mode == mode, report
    print("ok")
