# Environment
- Python 3.9 (analysis, tables), Python 3.12 venv for IFEval scoring (regex, packaging, langdetect, immutabledict, absl-py, nltk).
- Analysis: numpy, pandas, pyarrow, statsmodels, scipy.
- Generation: vLLM on NVIDIA GPUs; training: the released OLMo-2 DPO recipe (open-instruct) and LoRA-DPO for Qwen3-8B (see code/harness/e5, notes in prereg/).
- Third-party (not bundled): multi-value (dialect transforms) and llm-convergence (convergence baselines), fetched from their public repositories.
- LaTeX: tectonic, or pdfLaTeX + BibTeX (Overleaf).
