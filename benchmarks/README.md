# Benchmarks

These scripts reproduce the results in `docs/BENCHMARK.md` and `docs/KAGGLE_BENCHMARK.md`.
They are **not part of the library** and, unlike everything under `smartgate/` and `examples/`,
they need third-party tools:

| Needs | For |
|---|---|
| `numpy` | the Quora / Twitter metrics in `kaggle_benchmark.py` |
| `llama-cpp-python` + GGUF model files | the neural embedder and the local models used as small / medium / large tiers |
| the datasets (downloaded from Hugging Face / Kaggle; not redistributed) | the real traffic |

```bash
pip install numpy llama-cpp-python
python benchmarks/support_bot_benchmark.py --data <bitext.csv> --models-dir <models> --part all
python benchmarks/kaggle_benchmark.py --quora-csv questions.csv --twitter-csv twcs.csv --models-dir <models> --part all
```
