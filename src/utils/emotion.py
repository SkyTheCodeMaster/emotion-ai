from __future__ import annotations

import asyncio
import gc
import tomllib
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if TYPE_CHECKING:
  pass

with open("config.toml") as f:
  config = tomllib.loads(f.read())


MODEL_ID = config["ai"]["model"]
DEVICE = config["ai"]["device"]
MODEL_LOCK = asyncio.Lock()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

if not torch.cuda.is_available() and "cuda" in DEVICE:
  raise Exception("Cuda is not available for model inference!")


def classify(text):
  inputs = tokenizer(text, return_tensors="pt")
  outputs = model(**inputs)
  probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
  return [
    {"label": label, "score": prob.item()} for label, prob in enumerate(probs)
  ]


async def detect_emotion(text: str) -> dict:
  loop = asyncio.get_running_loop()
  async with MODEL_LOCK:
    result = await loop.run_in_executor(None, classify, text)
  cleanup()
  return result


def cleanup():
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()
  gc.collect()
