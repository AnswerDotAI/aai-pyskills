"""One-shot completion helpers for aai-pyskills, over fastllm.

`ask` runs a one-shot completion via `fastllm`: pass a `fastllm` short name or any full `'vendor/model'` spec, and get the response text back.
"""

__all__ = ['response_text', 'ask']

from fastllm.acomplete import acomplete
from fastllm.chat import contents, mk_msgs

def response_text(res):
    "The visible text of an `acomplete` result: its message's text parts joined (thinking parts skipped)."
    m = contents(res)
    return '' if not m else ''.join(p.text for p in m.content if p.type == 'text' and p.text)


async def ask(
    prompt,  # The user message
    model='sonn',  # A `fastllm` short name, or any full 'vendor/model' spec
    system=None,  # Optional system prompt
    effort=None,  # Reasoning effort where the model supports it: 'low'/'medium'/'high'
    max_tokens=8192,  # Response cap; fastllm's own defaults are small
    **kwargs,  # Passed through to `acomplete`
):
    "One-shot completion returning the response text."
    res = await acomplete(mk_msgs([prompt]), model, system=system, reasoning_effort=effort, max_tokens=max_tokens, **kwargs)
    return response_text(res)
