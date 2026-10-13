#!/usr/bin/env python
"""Merge a PEFT LoRA adapter into its bf16 base and save (weights + tokenizer).  python merge_lora.py --base BASE --adapter ADIR --out OUT"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
ap = argparse.ArgumentParser(); ap.add_argument("--base"); ap.add_argument("--adapter"); ap.add_argument("--out"); a = ap.parse_args()
m = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16); m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
m.save_pretrained(a.out, safe_serialization=True); AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out); print("merged ->", a.out)
