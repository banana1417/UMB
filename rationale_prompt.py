"""Prompt and parser for generating the two Qwen3-VL rationales."""

RATIONALE_PROMPT = """You are a multimodal fact-checking assistant. Given a news title and its accompanying image, analyze whether the text and image provide evidence that supports or weakens the credibility of the post.

Please generate two short rationales:
True reason: Explain why the post might be credible. Consider whether the image is consistent with the text, whether the visual content supports the claim, and whether the wording appears factual or contextually plausible.
False reason: Explain why the post might be misleading or fake. Consider possible image-text contradictions, reused or out-of-context images, visual manipulation artifacts, exaggerated or ambiguous wording, missing context, or other suspicious evidence.

Requirements:
1. Do not make a final true/false prediction.
2. Do not simply repeat the title.
3. Use both textual and visual evidence when possible.
4. If the evidence is insufficient, mention the uncertainty.
5. Keep each rationale concise and evidence-focused.

Output format:
True reason:
False reason:
"""


def parse_rationales(text):
    text = text.strip()
    lower = text.lower()
    true_key = "true reason:"
    false_key = "false reason:"
    true_start = lower.find(true_key)
    false_start = lower.find(false_key)
    if true_start < 0 or false_start < 0 or false_start <= true_start:
        raise ValueError("VLM output does not contain the required rationale labels")
    true_reason = text[true_start + len(true_key):false_start].strip()
    false_reason = text[false_start + len(false_key):].strip()
    return true_reason, false_reason
