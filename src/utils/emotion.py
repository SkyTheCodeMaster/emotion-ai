from __future__ import annotations

import asyncio
import gc
import tomllib
from typing import TYPE_CHECKING

import torch
from transformers import pipeline

if TYPE_CHECKING:
  from transformers import TextClassificationPipeline

with open("config.toml") as f:
  config = tomllib.loads(f.read())


MODEL_ID = config["ai"]["model"]
DEVICE = config["ai"]["device"]
MODEL_LOCK = asyncio.Lock()
pipe: TextClassificationPipeline = pipeline("text-classification", model=MODEL_ID, device=DEVICE)

if not torch.cuda.is_available() and "cuda" in DEVICE:
  raise Exception("Cuda is not available for model inference!")


def get_output(text: str) -> dict:
  if len(text) > 512:
    chunks: list[str] = [text[i:i+512] for i in range(0, len(text), 512)]
    results: list[list[dict[str,float]]] = []
    for chunk in chunks:
      results.append(pipe(chunk, top_k=999))
    
    summed_output: dict[str,tuple[float,int]] = {}
    for result in results:
      for data in result:
        label = data["label"]
        value = data["score"]
        if label in summed_output:
          summed_output[label] = (summed_output[label][0] + ((value-summed_output[label][0])/summed_output[label][1] + 1), summed_output[label][1] + 1)
        else:
          summed_output[label] = (value, 1) 
    output: dict[str,float] = {}
    for label, data in summed_output.items():
      output[label] = data[0]
    return output
  else:
    result = pipe(text, top_k=999)
    return result


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
