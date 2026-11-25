# aai-pyskills

Reusable [pyskills](https://github.com/AnswerDotAI/pyskills) for Answer.AI coding, writing, understanding, and vision workflows across CLI harnesses and Solveit dialogs.

## Install

```bash
pip install aai-pyskills
```

In a Python kernel with `pyskills` loaded, run `list_pyskills()` to list registered skills and `doc(aai_pyskills.<module>)` to read one.

## Modules

- `coding_patterns`: Python, Rust, nbdev, and testing conventions.
- `nbreview`: notebook style review workflow built on `nbdev.skill` and `coding_patterns`.
- `write_prose`: anti-slop rules for narrative prose, plus optional `check_prose`.
- `write_docs`: plain reference prose rules for docstrings, READMEs, API docs, PRs, and commits, plus optional `check_docs`.
- `write_summary`: standalone summaries for human readers.
- `write_llms`: dense documentation rules for LLM-facing files (`llms.txt`, pyskills, prompts).
- `understand`: building and testing a Theory (Naur, Socrates, Knoblich et al., Lakatos) before design or review work.
- `vision`: image and PDF inspection (`look`, `transcribe`, `pdf2pngs`) keeping visual content out of the caller's context.
- `llms`: short model names (`models`) and one-shot `ask()` helper used by `write_prose`, `write_docs`, and `vision`.

## Solveit and harness LLM calls

`llms.ask` and `vision.look`/`transcribe` check `JUPYGATE_LAUNCHER == 'rustygate'`. Inside a Solveit kernel, requests route over the `rustygate` comm via `dialoghelper` (`/api/prompt`) using the prompting user's Solve-LP access without direct provider API keys. Outside Solveit, requests go through `fastllm`.

## Development

```bash
pytest -q
```

Live model tests are marked `slow` and skipped by default (`pytest -q -m slow` runs them).
