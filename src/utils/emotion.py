from __future__ import annotations

import asyncio
import tomllib
from transformers import pipeline
import torch
import gc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  pass

with open("config.toml") as f:
  config = tomllib.loads(f.read())


MODEL_ID = config["ai"]["model"]
DEVICE = config["ai"]["device"]
MODEL_LOCK = asyncio.Lock()
pipe = pipeline("text-classification", model=MODEL_ID, device=DEVICE)

if not torch.cuda.is_available() and "cuda" in DEVICE:
  raise Exception("Cuda is not available for model inference!")

def get_output(text: str) -> dict:
  result = pipe(text)
  first = result[0]
  return {"emotion": first["label"], "confidence": first["score"]}

async def detect_emotion(text: str) -> dict:
  loop = asyncio.get_running_loop()
  async with MODEL_LOCK:
    result = await loop.run_in_executor(None, get_output, text)
  cleanup()
  return result

def cleanup():
  torch.cuda.empty_cache()
  torch.cuda.ipc_collect()
  gc.collect()