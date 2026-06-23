import sys
sys.path.append('/home/ubuntu/AcquisitionSynthesis/')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from typing import List, Dict
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
import datasets
from sklearn.cluster import MiniBatchKMeans
import services.service_utils as service_utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
import gc
import pandas as pd

os.environ['OPENBLAS_NUM_THREADS'] = '1'

API_KEY = os.getenv("CONF_API_KEY", "")
SERVER_IP = os.environ['SERVER_IP']

# vllm
language_model, llm_sampling_params = None, None

# vllm embedding
embedding_model = None
cluster_model: MiniBatchKMeans | None = None
cluster_centers_tensor: torch.Tensor | None = None  # shape: (100, embedding_dim)
NUM_CLUSTERS = 100

# gradient automodel
language_auto_model: AutoModelForCausalLM = None
language_tokenizer: AutoTokenizer = None

class ActivateRequest(BaseModel):
    service: str
    kwargs: Dict = Field(default_factory=dict)

class ActivateResponse(BaseModel):
    status: str
    service: str

class RewardsRequest(BaseModel):
    data: Dict

class RewardsResponse(BaseModel):
    acquisition_reward: float

# def cluster_proximity(embedding_model, embedding_dataset_name):
#     ds = pd.read_parquet(embedding_dataset_name, engine='pyarrow')
#     texts = list(ds.apply(lambda row: row['extra_info']['grounding_question'][0], axis=1))

#     print(f"Embedding {len(texts)} documents...")
#     embeddings = embedding_model.embed(texts)
#     embeddings = np.array([e.outputs.embedding for e in embeddings])

#     # Cluster into NUM_CLUSTERS clusters using MiniBatchKMeans (fast on large N)
#     print(f"Clustering into {NUM_CLUSTERS} clusters...")
#     cluster_model = MiniBatchKMeans(
#         n_clusters=NUM_CLUSTERS,
#         random_state=42,
#         batch_size=1028,
#         n_init=5,
        
#     )
#     cluster_model.fit(embeddings)

#     # Compute the mean embedding per cluster (L2-normalized for cosine similarity)
#     labels = cluster_model.labels_  # shape: (N,)
#     D = embeddings.shape[1]
#     centers = np.zeros((NUM_CLUSTERS, D), dtype=np.float32)
#     counts = np.zeros(NUM_CLUSTERS, dtype=np.int32)

#     for i, label in enumerate(labels):
#         centers[label] += embeddings[i]
#         counts[label] += 1

#     # Avoid division by zero for any empty cluster
#     counts = np.maximum(counts, 1)
#     centers = centers / counts[:, None]

#     # L2-normalize cluster centers so cosine sim = dot product later
#     norms = np.linalg.norm(centers, axis=1, keepdims=True)
#     norms = np.maximum(norms, 1e-8)
#     centers = centers / norms

#     cluster_centers_tensor = torch.tensor(centers)  # shape: (100, D)
#     print("Clustering complete.")
#     return cluster_centers_tensor

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    yield

app = FastAPI(lifespan=lifespan)

def clean_up():
    global language_model, llm_sampling_params, embedding_model, cluster_centers_tensor, language_auto_model, language_tokenizer
    if language_model is not None:
        language_model.llm_engine.engine_core.shutdown()
        destroy_model_parallel()
        destroy_distributed_environment()
        del language_model
    if embedding_model is not None:
        embedding_model.llm_engine.engine_core.shutdown()
        destroy_model_parallel()
        destroy_distributed_environment()
        del embedding_model
    if language_auto_model is not None:
        del language_auto_model
    if language_tokenizer is not None:
        del language_tokenizer

    language_model = None
    llm_sampling_params = None
    embedding_model = None
    cluster_centers_tensor = None
    language_tokenizer = None
    language_auto_model = None
    gc.collect()
    torch.cuda.empty_cache()
############################################# START SERVICE #############################################
@app.post("/start_service", response_model=ActivateResponse)
def start_service(req: ActivateRequest):
    # clean_up()
    global language_model, llm_sampling_params #, embedding_model, cluster_centers_tensor, language_auto_model, language_tokenizer
    
    if "mcot" in req.service:
        if language_model is not None:
            return {"status": "ok", "service": req.service}

        model_name = req.kwargs.get("model_name")
        language_model = LLM(model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9, trust_remote_code=True, max_model_len=4096)
        llm_sampling_params = SamplingParams(temperature=0.7, logprobs=2, max_tokens=512)
        service_utils.init_mcot_worker(language_model, llm_sampling_params)
        return {"status": "ok", "service": req.service}
    
    # if "confidence" in req.service:
    #     model_name = req.kwargs.get("model_name")
    #     language_model = LLM(model_name, tensor_parallel_size=1, gpu_memory_utilization=0.9, trust_remote_code=True, max_model_len=4096)
    #     llm_sampling_params = SamplingParams(temperature=0.7, logprobs=2, max_tokens=512)
    #     service_utils.init_confidence_worker(language_model, llm_sampling_params)
    #     return {"status": "ok", "service": req.service}

    # if "diversity" in req.service or "proximity" in req.service:
    #     model_name = "Qwen/Qwen3-Embedding-0.6B"
    #     dataset_name = req.kwargs.get("dataset_name")
    #     embedding_model = LLM(model_name, task="embed")
    #     cluster_centers_tensor = cluster_proximity(embedding_model, dataset_name)
    #     service_utils.init_proximity_worker(embedding_model, cluster_centers_tensor)
    #     service_utils.init_diversity_worker(embedding_model, cluster_centers_tensor)
    #     return {"status": "ok", "service": req.service}

    # if "gradient" in req.service:
    #     model_name = req.kwargs.get("model_name")
    #     language_auto_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    #     language_tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     language_tokenizer.pad_token_id = language_tokenizer.eos_token_id
    #     language_tokenizer.padding_size = "left"
    #     service_utils.init_gradient_worker(language_tokenizer, language_auto_model)
    #     return {"status": "ok", "service": req.service}

    # if "answer_variance" in req.service:
    #     model_name = req.kwargs.get("model_name")
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    #     language_model = LLM(model_name, tensor_parallel_size=1, gpu_memory_utilization=0.7, trust_remote_code=True)
    #     llm_sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
    #     os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    #     embedding_model = LLM("Qwen/Qwen3-Embedding-0.6B", task="embed")
    #     service_utils.init_answer_variance_worker(language_model, llm_sampling_params, embedding_model)
    #     return {"status": "ok", "service": req.service}

    
    if "format" in req.service:
        return {"status": "ok", "service": req.service}
    
    0/0
            
############################################# START SERVICE #############################################

############################################# CONFIDENCE #############################################
@app.post("/mcot", response_model=RewardsResponse)
def mcot_rewards(req: RewardsRequest, x_api_key: str | None = Header(default=None)):
    return service_utils.mcot(req)

# @app.post("/confidence", response_model=RewardsResponse)
# def conf_rewards(req: RewardsRequest, x_api_key: str | None = Header(default=None)):
#     return service_utils.confidence(req)

# @app.post("/diversity", response_model=RewardsResponse)
# def diversity_rewards(req: RewardsRequest, x_api_key: str | None = Header(default=None)):
#     return service_utils.diversity(req)

# @app.post("/proximity", response_model=RewardsResponse)
# def proximity_rewards(req: RewardsRequest, x_api_key: str | None = Header(default=None)):
#     return service_utils.proximity(req)

# @app.post("/gradient", response_model=RewardsResponse)
# def gradient_rewards(req: RewardsRequest, x_api_key: str | None = Header(default=None)):
#     return service_utils.gradient(req)

# @app.post("/answer_variance", response_model=RewardsResponse)
# def answer_variance_rewards(req: RewardsRequest, x_api_key: str | None = Header(default=None)):
#     return service_utils.answer_variance(req)
############################################# ANSWER_VARIANCE #############################################

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("all:app", host=SERVER_IP, port=5145)
