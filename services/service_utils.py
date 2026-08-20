import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import threading
import queue
import uuid
import torch
from pydantic import BaseModel
import hdbscan


class RewardsResponse(BaseModel):
    acquisition_reward: float


class WorkerQueue:
    """Serializes GPU calls through a single background worker thread, with dynamic batching."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._results: dict = {}
        self._events: dict = {}
        self._lock = threading.Lock()

    def start(self, fn, *args):
        t = threading.Thread(target=self._worker, args=(fn, args), daemon=True)
        t.start()

    def _worker(self, fn, args):
        while True:
            # Block until at least one item is ready
            job_id, req = self._queue.get()
            batch = [(job_id, req)]
            # Drain any additional pending items without blocking
            while True:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            try:
                results = fn([req for _, req in batch], *args)
            except Exception:
                results = [RewardsResponse(acquisition_reward=-5.0)] * len(batch)

            with self._lock:
                for (jid, _), result in zip(batch, results):
                    self._results[jid] = result
                    self._events[jid].set()

    def submit(self, req) -> RewardsResponse:
        job_id = str(uuid.uuid4())
        event = threading.Event()
        with self._lock:
            self._events[job_id] = event
        self._queue.put((job_id, req))
        event.wait()
        with self._lock:
            result = self._results.pop(job_id)
            del self._events[job_id]
        return result


# --- module-level queues (one per service) ---
_confidence_queue: WorkerQueue | None = None
_gradient_queue: WorkerQueue | None = None
_proximity_queue: WorkerQueue | None = None
_diversity_queue: WorkerQueue | None = None
_answer_variance_queue: WorkerQueue | None = None

# Running facility-location ground set for diversity, built purely from the stream
# of generated samples (not the pre-existing dataset). _diversity_archive holds
# kept sample embeddings; _diversity_best_d[i] is the largest distance
# (1 - cosine sim) any later sample has achieved from _diversity_archive[i] (each
# entry starts at 0, its own self-distance). Only ever touched from the single
# diversity worker thread, so mutation without a lock is safe. Reset in
# init_diversity_worker on service (re)start.
_diversity_archive: torch.Tensor | None = None
_diversity_best_d: torch.Tensor | None = None

# Minimum marginal gain a sample must produce to be admitted to the ground set.
# Without this, a near-duplicate still nudges some archive member's best_d by an
# infinitesimal amount and would count as "updating the frontier," so the archive
# would never actually get pruned. Keeps archive size (and per-sample compute)
# bounded over a long run.
_DIVERSITY_MIN_GAIN_TO_ADMIT = 0.01


# --- pure compute functions ---

def _compute_confidence(reqs, language_model, sampling_params):
    questions = [req.data['question'] for req in reqs]
    outputs = language_model.generate(questions, sampling_params=sampling_params)
    results = []
    for output in outputs:
        logprobs = output.outputs[0].logprobs
        top1, top2 = [], []
        for lp in logprobs:
            keys = list(lp.keys())
            if len(keys) >= 2:
                top1.append(lp[keys[0]].logprob)
                top2.append(lp[keys[1]].logprob)
        if not top1:
            results.append(RewardsResponse(acquisition_reward=-5.0))
            continue
        avg_diff = (torch.exp(torch.tensor(top1)) - torch.exp(torch.tensor(top2))).mean()
        results.append(RewardsResponse(acquisition_reward=float(1.0 / avg_diff)))
    return results


def _compute_proximity(reqs, embedding_model, cluster_centers_tensor):
    questions = [req.data['question'] for req in reqs]
    completion_embeddings = embedding_model.embed(questions)
    completion_tensor = torch.Tensor([c.outputs.embedding for c in completion_embeddings])
    completion_tensor = completion_tensor.to(cluster_centers_tensor.dtype)
    similarities = completion_tensor @ cluster_centers_tensor.T  # (B, 100)
    nearest_cluster_ids = similarities.argmax(dim=1)
    return [
        RewardsResponse(acquisition_reward=similarities[i, cid].item())
        for i, cid in enumerate(nearest_cluster_ids)
    ]


def _compute_diversity(reqs, embedding_model):
    """
    facility location esque
    """
    global _diversity_archive, _diversity_best_d

    questions = [req.data['question'] for req in reqs]
    completion_embeddings = embedding_model.embed(questions)
    completion_tensor = torch.Tensor([c.outputs.embedding for c in completion_embeddings])

    results = []
    for i in range(len(reqs)):
        x = completion_tensor[i]

        if _diversity_archive is None:
            gain, admit = 1.0, True
        else:
            sims = _diversity_archive @ x  # (N,)
            d = 1.0 - sims
            gain = torch.clamp(d - _diversity_best_d, min=0.0).mean().item()
            admit = gain > _DIVERSITY_MIN_GAIN_TO_ADMIT
            if admit:
                _diversity_best_d = torch.maximum(_diversity_best_d, d)

        results.append(RewardsResponse(acquisition_reward=float(gain)))

        if admit:
            x_row, self_d = x.unsqueeze(0), torch.tensor([0.0], dtype=x.dtype)
            if _diversity_archive is None:
                _diversity_archive, _diversity_best_d = x_row, self_d
            else:
                _diversity_archive = torch.cat([_diversity_archive, x_row], dim=0)
                _diversity_best_d = torch.cat([_diversity_best_d, self_d], dim=0)

    return results


def _compute_gradient(reqs, tokenizer, model):
    # Gradient must be computed per-sample (backward pass resets grads), so loop
    results = []
    device = next(model.parameters()).device
    for req in reqs:
        prompt = req.data['question']
        output = req.data['answer']

        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True)

        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        labels = input_ids.clone()
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}],
            tokenize=False, add_generation_prompt=False
        )
        prompt_len = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
        labels[:, :prompt_len] = -100

        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        loss.backward()

        total_sq = sum(
            p.grad.detach().float().norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        )
        results.append(RewardsResponse(acquisition_reward=total_sq ** 0.5))
    return results


def _compute_answer_variance(reqs, language_model, sampling_params, embedding_model, k=16):
    # Generate k samples per question in one batched call
    questions_repeated = [req.data['question'] for req in reqs for _ in range(k)]
    raw_outputs = language_model.generate(questions_repeated, sampling_params=sampling_params)
    texts = [o.outputs[0].text.strip() for o in raw_outputs]

    all_embeddings = embedding_model.embed(texts)
    all_embeddings = torch.Tensor([a.outputs.embedding for a in all_embeddings])

    results = []
    for i in range(len(reqs)):
        chunk = all_embeddings[i * k:(i + 1) * k]
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
        labels = clusterer.fit_predict(chunk)
        results.append(RewardsResponse(acquisition_reward=float(len(set(labels)))))
    return results


# --- init functions (called from all.py on service start) ---

def init_confidence_worker(language_model, sampling_params):
    global _confidence_queue
    _confidence_queue = WorkerQueue()
    _confidence_queue.start(_compute_confidence, language_model, sampling_params)

def init_gradient_worker(tokenizer, model):
    global _gradient_queue
    _gradient_queue = WorkerQueue()
    _gradient_queue.start(_compute_gradient, tokenizer, model)

def init_proximity_worker(embedding_model, cluster_centers_tensor):
    global _proximity_queue
    _proximity_queue = WorkerQueue()
    _proximity_queue.start(_compute_proximity, embedding_model, cluster_centers_tensor)

def init_diversity_worker(embedding_model):
    global _diversity_queue, _diversity_archive, _diversity_best_d
    _diversity_queue = WorkerQueue()
    _diversity_archive = None
    _diversity_best_d = None
    _diversity_queue.start(_compute_diversity, embedding_model)

def init_answer_variance_worker(language_model, sampling_params, embedding_model):
    global _answer_variance_queue
    _answer_variance_queue = WorkerQueue()
    _answer_variance_queue.start(_compute_answer_variance, language_model, sampling_params, embedding_model)


# --- public API (called from all.py routes) ---

def confidence(req):
    if _confidence_queue is None:
        raise RuntimeError("confidence worker not initialized — call /start_service first")
    return _confidence_queue.submit(req)

def gradient(req):
    if _gradient_queue is None:
        raise RuntimeError("gradient worker not initialized — call /start_service first")
    return _gradient_queue.submit(req)

def proximity(req):
    if _proximity_queue is None:
        raise RuntimeError("proximity worker not initialized — call /start_service first")
    return _proximity_queue.submit(req)

def diversity(req):
    if _diversity_queue is None:
        raise RuntimeError("diversity worker not initialized — call /start_service first")
    return _diversity_queue.submit(req)

def answer_variance(req):
    if _answer_variance_queue is None:
        raise RuntimeError("answer_variance worker not initialized — call /start_service first")
    return _answer_variance_queue.submit(req)
