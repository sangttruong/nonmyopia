"""
This is reward model server main process.
"""

import argparse
import gc
from contextlib import asynccontextmanager
from typing import List

import torch
import uvicorn
from datasets import Dataset
from embed_text_package.embed_text import Embedder
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch.utils.data import DataLoader

from transformers.utils import (
    is_peft_available,
    is_torch_cuda_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()


def torch_gc() -> None:
    r"""
    Collects GPU or NPU memory.
    """
    gc.collect()
    if is_torch_xpu_available():
        torch.xpu.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_cuda_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(_: "FastAPI"):
    """
    Collects GPU memory
    """
    yield
    torch_gc()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedder = Embedder()
embedder.load(args.model, special_tokens=True)


class EmbResponse(BaseModel):
    """
    Response data class
    """

    embedding: List[List[float]] = []


class EmbRequest(BaseModel):
    """
    Request data class
    """

    messages: List[str]


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


@app.post(
    "/",
    response_model=EmbResponse,
    status_code=status.HTTP_200_OK,
)
async def compute_emb(request: EmbRequest):
    """
    This function is used to call reward model for computing reward scores.
    """
    ds = Dataset.from_dict({"text": request.messages})
    ds_emb = (
        embedder.get_embeddings(
            DataLoader(ds, batch_size=args.batch_size),
            embedder.which_model,
            ["text"],
        )
        .data["text"]
        .to_pylist()
    )

    return EmbResponse(embedding=ds_emb)


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheck: Returns a JSON response with the health status
    """
    return HealthCheck(status="OK")


def run_server():
    """
    Main function for starting reward server
    """
    print(f"Visit http://localhost:{args.port}/docs for API document.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run_server()
