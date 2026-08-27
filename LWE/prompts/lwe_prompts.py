"""
Prompts for sample-specific evaluation (SSP) and learning-with-experience (LWE).

Aligned with the paper’s meta-prompt / meta-eval / batch-update design.
"""

INITIAL_META_PROMPT = """Generate a prompt for an evaluator that includes example-specific evaluation criteria and step-by-step evaluation procedures. The evaluator will base their evaluation on the prompt and it will help the evaluator accurately judge the correctness and reasoning quality of given responses, and identify which answer is better.

Generated evaluation prompt MUST include
- evaluation criteria specific for the given example
- evaluation steps specific for the given example to induce an accurate and reliable judgment.

Additionally, the evaluation prompt MUST instruct the evaluator that their final judgment must be expressed only in one of the following two formats:
'[[A]]' or '[[B]]'.

Output exactly ONLY THE PROMPT."""

STATIC_REQUIREMENTS_FOR_EVAL_PROMPT_GENERATION = """NEVER judge the example contents.
ALWAYS generate evaluation criteria and evaluation steps based on the example.
NEVER put the example contents in the generated evaluation prompt. You MUST generate only the template for the evaluation prompt.
NEVER include the given meta prompt in your output."""

STATIC_REQUIREMENTS_FOR_EVAL_PROMPT = """The final judgment MUST be expressed only in one of the following two formats:
'[[A]]' or '[[B]]'."""

STATIC_EXAMPLE_PLACEHOLDER = """[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"""

STATIC_JUDGMENT_PLACEHOLDER = """[The Start of Judgment]\n{judgment}\n[The End of Judgment]"""

STATIC_META_FEEDBACK_PLACEHOLDER = """[The Start of Meta Feedback]\n{meta_feedback}\n[The End of Meta Feedback]"""

STATIC_PROMPT_FOR_META_FEEDBACK = """You will be given:
An evaluation prompt — the instructions or criteria for judging
A given example — a case where you have already made a judgment based on that prompt

Your task:
Evaluate the correctness and reasoning quality of the given judgment.
Based on the inspection, update your [[Learned tips for future prompt optimization]] based on the current "meta_prompt".

Your output must have three parts:
[[[Score (1–5)]]]
5 = Judgment is entirely correct, thorough, and follows the evaluation prompt exactly
4 = Mostly correct, with only minor reasoning gaps
3 = Partially correct, but with noticeable flaws or missing justification
2 = Largely incorrect, due to poor reasoning or serious omissions
1 = Fundamentally wrong, fails to follow the evaluation prompt

[[[Binary label]]]
"Absolutely confident the judgment is correct" — if reasoning is strong, evidence-based, and follows the criteria without major gaps
"Not sure" — if there are reasoning flaws, possible alternative interpretations, or insufficient evidence

[[[Learned tips for future prompt optimization]]]
After reading the given example and its judgment, identify specific points to improve in the reasoning, coverage, or clarity.
Reflect on these findings to propose concrete adjustments to future evaluation prompts (e.g., clarifying ambiguous criteria, adding explicit checks for evidence, requiring certain logical structures)
Capture any patterns that might cause repeated errors or inconsistencies so they can be addressed in later prompts.
You might refer to the current meta prompt and identify additional essential tips that are not addressed in the current meta prompt.

**Evaluation Steps (You must follow these before scoring)**:
Step 1: Restate the evaluation criteria in your own words to ensure you understand them.
Step 2: Read the given example judgment carefully and identify its main claim(s) and supporting reasoning.
Step 3: Compare the judgment's reasoning with the evaluation criteria; look for matches, missing points, contradictions, or misinterpretations.
Step 4: Inspect for logical fallacies, including:
    - Unsupported generalizations
    - False cause (confusing correlation with causation)
    - Strawman misrepresentations of the criteria
    - Circular reasoning
    - Irrelevant evidence
Step 5: Pinpoint exact areas for improvement in the judgment's reasoning or evidence use.
Step 6: Assign a score based on the completeness and accuracy of the reasoning relative to the prompt.
Step 7: Decide on the binary confidence label.
Step 8: Reflect on how these identified weaknesses could be prevented through clearer or more detailed future evaluation prompts, and write down 1–3 learned tips.

[[[The Start of Current Meta Prompt]]]
{meta_prompt}
[[[The End of Current Meta Prompt]]]

[[[The Start of Evaluation Prompt & Judgment]]]
{evaluation_prompt}

{static_judgment_place_holder}
[[[The End of Evaluation Prompt & Judgment]]]

Strictly follow the **Output format**:
{{"score": score, "label": label, "learned tips": tips, "reasoning": reasoning_or_explanation}}"""

STATIC_PROMPT_FOR_META_PROMPT_UPDATE = """Optimize your current meta prompt. You need to refer to feedback. You need to improve the meta prompt. The goal of meta prompt is to generate an evaluation prompt that helps more accurate and reliable judgments. A good evaluation prompt needs better example-specific evaluation rubrics and evaluation steps. Make sure not to repeat any tips that are already provided. Instead, write only additional tips that are general, reusable, and useful for evaluation. Focus on refining and improving the quality of your learned tips and experiences.

[[[The Start of Current Meta Prompt]]]
{meta_prompt}
[[[The End of Current Meta Prompt]]]

[[[The Start of Batch examples & Meta Feedback]]]
{batch}
[[[The End of Batch examples & Meta Feedback]]]

Again, optimize your current meta prompt based on the feedback.
[[[Optimized Meta Prompt]]]
"""

STATIC_PROMPT_FOR_META_PROMPT_UPDATE_RESTRICT_LENGTH = """Optimize your current meta prompt. You need to refer feedback. You need to improve the meta prompt. The goal of meta prompt is to generate an evaluation prompt that helps more accurate and reliable judgments. A good evaluation prompt needs better example-specific evaluation rubrics and evaluation steps.

[[[The Start of Current Meta Prompt]]]
{meta_prompt}
[[[The End of Current Meta Prompt]]]

[[[The Start of Batch examples & Meta Feedback]]]
{batch}
[[[The End of Batch examples & Meta Feedback]]]

Again, optimize your current meta prompt based on the feedback.
**If your meta prompt is too long, you need to shorten it.**
[[[Optimized Meta Prompt]]]
"""

SUMMARIZE_META_PROMPT = """You are a helpful assistant that summarizes and shortens text. Your current meta prompt is too long.
[[[The Start of Current Meta Prompt]]]
{meta_prompt}
[[[The End of Current Meta Prompt]]]

Condense the meta prompt so it is roughly HALF as long, but still structured and keeping the key details.
ONLY output the shortend meta prompt.
"""
