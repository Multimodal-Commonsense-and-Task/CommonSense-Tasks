"""Central prompt configuration for the BINDER agent.

All natural-language prompts that steer the two modules live here so they can be
edited in one clean place:

  * DRM_GUIDANCE_OUTPUT_INSTRUCTION  - appended to the DRM planning prompt so the DRM
    additionally emits a task-specific *guidance prompt* G (paper Eq. 1).
  * DEFAULT_GUIDANCE                  - fallback guidance G per action phase, used when
    the DRM does not supply one.
  * compose_irm_prompt()             - builds the prompt actually sent to the IRM
    Video-LLM = guidance G (the "what to focus on" part) + a fixed response-format
    template (the "how to answer" part, get_irm_response_format() below).

  * get_irm_response_format()        - the fixed response-format templates the IRM
    Video-LLM must answer in. These define the exact output schema that the
    guidance-conditioned parser (Phi_G, see guidance_parser.py) expects, so edit them
    together with guidance_parser.py.

This is the single place to edit every agent prompt.
"""

# --------------------------------------------------------------------------- #
# (1) DRM guidance output — paper Eq. 1: (a, o, G) = DRM(P, M)                  #
# --------------------------------------------------------------------------- #
# Appended to the DRM planning prompt. Instructs the DRM to additionally produce a
# "guidance" field: a short natural-language prompt G that refocuses the IRM's attention
# for the upcoming action phase.
DRM_GUIDANCE_OUTPUT_INSTRUCTION = """
In addition to the action, also produce a "guidance" field: a short natural-language
instruction that tells the background Instance Response Module (IRM, a video monitor)
what to pay attention to during this action. The IRM continuously watches the robot's
camera and reports task-relevant changes.

- During navigation/exploration, the guidance should ask the IRM to alert when
  task-relevant objects or receptacles appear or move.
- During grasping/placing, the guidance should ask the IRM to watch gripper-object
  alignment and flag when the manipulation is likely to fail.

Keep the guidance to one or two sentences, focused on the CURRENT action and the
task-relevant objects.
"""

# Example "guidance" value shown to the DRM in the output-format block.
GUIDANCE_OUTPUT_EXAMPLE = (
    "Current action is navigation; alert me when the target object or any task "
    "receptacle becomes visible or appears to have moved."
)

# --------------------------------------------------------------------------- #
# (2) Default guidance G per action phase (fallback when DRM omits one).        #
#     Mirrors the phrasing in the paper (Fig. 3).                               #
# --------------------------------------------------------------------------- #
DEFAULT_GUIDANCE = {
    "goto_target_point": (
        "Current action is navigation. Alert me as soon as any task-relevant object "
        "or receptacle ({targets}) becomes visible or appears to have moved."
    ),
    "explore_environment": (
        "Current action is exploration. Alert me the moment the target ({targets}) or "
        "any task-relevant object first becomes visible."
    ),
    "_grasp": (
        "Current action is grasping {targets}. Watch the gripper-object alignment and "
        "flag immediately if the grasp is likely to fail."
    ),
    "_place": (
        "Current action is placing into {targets}. Watch object-receptacle alignment "
        "and flag if placement is likely to fail."
    ),
}


def get_default_guidance(action: str, target_objects=None) -> str:
    """Return the fallback guidance prompt G for an action phase."""
    targets = ", ".join(target_objects) if target_objects else "the task objects"
    template = DEFAULT_GUIDANCE.get(action)
    if template is None:
        return f"Monitor the scene for task-relevant objects ({targets})."
    return template.format(targets=targets)


# --------------------------------------------------------------------------- #
# (3) IRM response-format templates (the "how to answer" schema).              #
#     These define the exact output Phi_G (guidance_parser.py) parses.          #
# --------------------------------------------------------------------------- #
def get_irm_response_format(report_mode: str, target_objects) -> str:
    """Return the response-format template the IRM Video-LLM must answer in.

    'report_mode' is described in the context of the IRM, not the robot state: while the
    robot navigates to a localized target, the IRM still checks the other targets too.
    """
    if report_mode == "explore_multiple_targets":
        return """
You are given a short egocentric video from a robot.

Describe all objects visible in the video in a single sentence.
Do NOT include actions or locations. Just name visible objects.
e.g. "yellow cup, red box, apple"
"""
    if report_mode == "watch_grasping":
        # Free-form report (paper: Z_t is natural language). The guidance-conditioned
        # parser Phi_G (guidance_parser.py) decides the execution mode from this text.
        return f"""
You are watching a robot gripper that just attempted to grasp the {target_objects[0]}.
In one or two sentences, freely describe what you see between the gripper fingers:
is the {target_objects[0]} held there, is it slipping or being dropped, and does the
grasp look successful or failed?"""
    if report_mode == "grasping_refinement":
        return f"""
You are watching a robot arm preparing to grasp the {target_objects[0]}.
In one or two sentences, freely describe whether the gripper looks well-positioned to
grasp the {target_objects[0]}, or whether it seems misaligned or unable to reach it."""
    raise ValueError(f"Invalid report_mode: {report_mode}")


# --------------------------------------------------------------------------- #
# (4) Compose the prompt sent to the IRM Video-LLM.                             #
#     full prompt = guidance G (focus) + response-format template (schema).     #
# --------------------------------------------------------------------------- #
def compose_irm_prompt(guidance: str, report_mode: str, target_objects) -> str:
    """Build the IRM Video-LLM prompt.

    Args:
        guidance: task-specific guidance prompt G (from the DRM, or a default). If
            empty, only the response-format template is sent.
        report_mode: response-format key understood by get_irm_response_format
            (e.g. "explore_multiple_targets", "watch_grasping", "grasping_refinement").
        target_objects: objects relevant to the current phase.

    Returns:
        The full prompt string to send to the Video-LLM.
    """
    response_format = get_irm_response_format(report_mode, target_objects)
    if guidance:
        return f"{guidance.strip()}\n\n{response_format}"
    return response_format
