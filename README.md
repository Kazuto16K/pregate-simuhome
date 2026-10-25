# PreGate on SimuHome

A verification layer for LLM agents that schedule smart-home actions. A small language model (Qwen3-1.7B + LoRA)
turns the request into time constraints, Z3 proves contradictions or computes an allowed time window per action,
and a gate around the agent's `schedule_workflow` tool checks every call (time window, dry run on a copy of the
home) and refuses wrong calls with the fix. Evaluated on SimuHome QT4 (300 scheduling episodes).

This repository contains only the code and data behind the reported results (gate v2.2, Qwen2.5-14B agent).

## Layout

| Path | Content |
|---|---|
| `src/` | The method (`gate.py`, `grounding.py`, `constraint_encoder.py`, `shadow_sim.py`, ...) and the runner (`parallel_runner.py`, `pausable_sim.py`), analysis (`analysis.py`, `parse_eval.py`), rewording check (`rewording.py`) |
| `notebooks/01_train_parser_v3_cv.ipynb` | 5-fold training of the parser and the parse cache (GPU) |
| `notebooks/00_approach2_local_partA.ipynb`, `..._partB.ipynb` | The evaluation: agent and judge Qwen2.5-14B served with Ollama on Kaggle T4 x2; each part runs half of the episodes of every cell and arm |
| `notebooks/03_rewording_robustness.ipynb` | Parser robustness to reworded requests |
| `tests/` | Unit and integration tests; `replay_traces.py`, `diagnose_failures.py`, `eval_cv_cache.py` |
| `data/parser/` | 5-fold parse cache and parser evaluation (adapter weights not included) |
| `results/qwen2.5-14b/` | All episode records (SimuHome format), gate telemetry and run logs of both parts |
| `verify_results.py` | Recomputes every reported number from `data/` and `results/` and checks it |

## Setup

SimuHome (CC BY-NC-ND 4.0) is not included; clone it next to this repository and install it:

```bash
git clone --depth 1 https://github.com/holi-lab/SimuHome.git ../SimuHome
pip install -e ../SimuHome --ignore-requires-python
pip install -r requirements.txt
```

Scripts look for SimuHome in `../SimuHome` (or the path given as an argument / in `SIMUHOME_DIR`).

## Reproduce

```bash
python verify_results.py          # every reported number, from the stored data
python tests/test_pipeline.py     # Z3 encoder on the ground-truth labels (286/286, 301/301)
python tests/test_gate.py         # gate on all 300 homes, no LLM
python tests/test_e2e_fake_llm.py # real simulators, scripted LLM, clock hold
```

To re-run the evaluation: train the parser with notebook 01 (or use `data/parser/parse_cache_v3_cv.json`), then run
notebooks 00 part A and part B on Kaggle (GPU T4 x2, internet on) with the parse cache attached.
