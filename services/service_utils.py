import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

import threading
import queue
import uuid
import torch
from pydantic import BaseModel

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
_mcot_queue: WorkerQueue | None = None
# _confidence_queue: WorkerQueue | None = None
# _gradient_queue: WorkerQueue | None = None
# _proximity_queue: WorkerQueue | None = None
# _diversity_queue: WorkerQueue | None = None
# _answer_variance_queue: WorkerQueue | None = None


# --- pure compute functions ---
def calculate_confidence(outputs):
    confidence = []
    for output in outputs:
        logprobs = output.outputs[0].logprobs
        top1, top2 = [], []
        for lp in logprobs:
            keys = list(lp.keys())
            if len(keys) >= 2:
                top1.append(lp[keys[0]].logprob)
                top2.append(lp[keys[1]].logprob)
        if not top1:
            confidence.append(-5.0)
            continue
        avg_diff = (torch.exp(torch.tensor(top1)) - torch.exp(torch.tensor(top2))).mean()
        confidence.append(float(1.0 / avg_diff))
    return confidence

def _compute_mcot(reqs, language_model, sampling_params):
    questions = [req.data['question'] for req in reqs]
    answers = [req.data['answer'] for req in reqs]
    english_only_instruction = lambda question: (
        "Answer the following question. Reason step-by-step in English inside <reasoning> tags, then give your final answer inside <answer> tags.\n"
        f"<question>\n{question}\n</question>\n"
        "<reasoning>\n\n</reasoning>\n"
        "<answer>\n\n</answer>"
    )
    mcot_instruction = lambda question: (
        "Answer the following question. Write each reasoning step in a different language from: English, Spanish, French, Italian, or Portuguese. "
        "Example:\n<question>A café orders 4 boxes of croissants on Monday and 7 boxes on Tuesday. Each box costs $9. How much did the café spend in total?<question>\n"
        "<reasoning>The goal is to find the total amount spent across both days. Primero, sumamos las cajas de ambos días: 4 + 7 = 11 cajas en total. Chaque boîte coûte 9 $, donc il faut multiplier le nombre de boîtes par le prix unitaire. Quindi calcoliamo: 11 × 9 = 99. Observa que el número de piezas en cada caja no es necesario para calcular el costo total, sino solo el número de cajas y su precio. Portanto, o café gastou um total de 99 dólares.\n </reasoning>"
        "<answer> $99. </answer>"
        "Place all reasoning inside <reasoning> tags and your final answer in English inside <answer> tags.\n"
        f"<question> {question} </question>\n"
        "<reasoning> </reasoning>\n"
        "<answer> </answer>"
    )

    english_prompts = [english_only_instruction(q) for q in questions]
    english_outputs = language_model.generate(english_prompts, sampling_params=sampling_params)
    english_parsed_outputs = [o.outputs[0].text.strip() for o in english_outputs]
    
    mcot_prompts = [mcot_instruction(q) for q in questions]
    mcot_outputs = language_model.generate(mcot_prompts, sampling_params=sampling_params)
    mcot_parsed_outputs = [o.outputs[0].text.strip() for o in mcot_outputs]

    eng_confidence = calculate_confidence(english_outputs)
    mcot_confidence = calculate_confidence(mcot_outputs)

    rewards = []
    for eng, mcot, answer, eng_conf, mcot_conf in zip(english_parsed_outputs, mcot_parsed_outputs, answers, eng_confidence, mcot_confidence):
        
        # vV3
        if eng_conf - mcot_conf > 0.1: # confidence of mcot is significantly bad, we want to increase it, regardless of the answer
            rewards.append(1.0)
        elif abs(eng_conf - mcot_conf) <= 0.1: # pretty much the same confidence, it would be good if mcot is incorect while eng is correct:
            rewards.append(0.1)
        elif mcot_conf - eng_conf > 0.1: # mcot confidence is pretty good, regardless of correctness we don't need to touch these samples
            rewards.append(0.0)



        # V4
        # eng_correct = answer in eng
        # mcot_correct = answer in mcot

        # if eng_conf - mcot_conf > 0.1: # confidence of mcot is significantly bad, we want to increase it, regardless of the answer
        #     rewards.append(1.0)
        # elif abs(eng_conf - mcot_conf) <= 0.1: # pretty much the same confidence, it would be good if mcot is incorect while eng is correct:
        #     if eng_correct and not mcot_correct:
        #         rewards.append(1.0)
        #     else:
        #         rewards.append(0.0)
        # elif mcot_conf - eng_conf > 0.1: # mcot confidence is pretty good, regardless of correctness we don't need to touch these samples
        #     rewards.append(0.0)

        # V1/v2
        # if eng_correct and mcot_correct:
        #     rewards.append(0.0)
        # elif eng_correct and not mcot_correct:
        #     rewards.append(1.0)
        # elif not eng_correct and mcot_correct:
        #     rewards.append(0.25)
        # elif not eng_correct and not mcot_correct:
        #     rewards.append(0.5)


    return [RewardsResponse(acquisition_reward=r) for r in rewards]

# def _compute_confidence(reqs, language_model, sampling_params):
#     questions = [req.data['question'] for req in reqs]
#     outputs = language_model.generate(questions, sampling_params=sampling_params)
#     results = []
#     for output in outputs:
#         logprobs = output.outputs[0].logprobs
#         top1, top2 = [], []
#         for lp in logprobs:
#             keys = list(lp.keys())
#             if len(keys) >= 2:
#                 top1.append(lp[keys[0]].logprob)
#                 top2.append(lp[keys[1]].logprob)
#         if not top1:
#             results.append(RewardsResponse(acquisition_reward=-5.0))
#             continue
#         avg_diff = (torch.exp(torch.tensor(top1)) - torch.exp(torch.tensor(top2))).mean()
#         results.append(RewardsResponse(acquisition_reward=float(1.0 / avg_diff)))
#     return results


# def _compute_proximity(reqs, embedding_model, cluster_centers_tensor):
#     questions = [req.data['question'] for req in reqs]
#     completion_embeddings = embedding_model.embed(questions)
#     completion_tensor = torch.Tensor([c.outputs.embedding for c in completion_embeddings])
#     completion_tensor = completion_tensor.to(cluster_centers_tensor.dtype)
#     similarities = completion_tensor @ cluster_centers_tensor.T  # (B, 100)
#     nearest_cluster_ids = similarities.argmax(dim=1)
#     return [
#         RewardsResponse(acquisition_reward=similarities[i, cid].item())
#         for i, cid in enumerate(nearest_cluster_ids)
#     ]


# def _compute_diversity(reqs, embedding_model, cluster_centers_tensor):
#     questions = [req.data['question'] for req in reqs]
#     completion_embeddings = embedding_model.embed(questions)
#     completion_tensor = torch.Tensor([c.outputs.embedding for c in completion_embeddings])
#     completion_tensor = completion_tensor.to(cluster_centers_tensor.dtype)
#     similarities = completion_tensor @ cluster_centers_tensor.T  # (B, 100)
#     return [
#         RewardsResponse(acquisition_reward=float(1.0 - similarities[i].max().item()))
#         for i in range(len(reqs))
#     ]


# def _compute_gradient(reqs, tokenizer, model):
#     # Gradient must be computed per-sample (backward pass resets grads), so loop
#     results = []
#     device = next(model.parameters()).device
#     for req in reqs:
#         prompt = req.data['question']
#         output = req.data['answer']

#         messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}]
#         text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#         enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True)

#         input_ids = enc["input_ids"].to(device)
#         attention_mask = enc["attention_mask"].to(device)

#         labels = input_ids.clone()
#         prompt_text = tokenizer.apply_chat_template(
#             [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}],
#             tokenize=False, add_generation_prompt=False
#         )
#         prompt_len = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
#         labels[:, :prompt_len] = -100

#         model.zero_grad()
#         with torch.autocast("cuda", dtype=torch.bfloat16):
#             loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
#         loss.backward()

#         total_sq = sum(
#             p.grad.detach().float().norm().item() ** 2
#             for p in model.parameters() if p.grad is not None
#         )
#         results.append(RewardsResponse(acquisition_reward=total_sq ** 0.5))
#     return results


# def _compute_answer_variance(reqs, language_model, sampling_params, embedding_model, k=16):
#     # Generate k samples per question in one batched call
#     questions_repeated = [req.data['question'] for req in reqs for _ in range(k)]
#     raw_outputs = language_model.generate(questions_repeated, sampling_params=sampling_params)
#     texts = [o.outputs[0].text.strip() for o in raw_outputs]

#     all_embeddings = embedding_model.embed(texts)
#     all_embeddings = torch.Tensor([a.outputs.embedding for a in all_embeddings])

#     results = []
#     for i in range(len(reqs)):
#         chunk = all_embeddings[i * k:(i + 1) * k]
#         clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean")
#         labels = clusterer.fit_predict(chunk)
#         results.append(RewardsResponse(acquisition_reward=float(len(set(labels)))))
#     return results


# --- init functions (called from all.py on service start) ---

def init_mcot_worker(language_model, sampling_params):
    global _mcot_queue
    _mcot_queue = WorkerQueue()
    _mcot_queue.start(_compute_mcot, language_model, sampling_params)

# def init_confidence_worker(language_model, sampling_params):
#     global _confidence_queue
#     _confidence_queue = WorkerQueue()
#     _confidence_queue.start(_compute_confidence, language_model, sampling_params)

# def init_gradient_worker(tokenizer, model):
#     global _gradient_queue
#     _gradient_queue = WorkerQueue()
#     _gradient_queue.start(_compute_gradient, tokenizer, model)

# def init_proximity_worker(embedding_model, cluster_centers_tensor):
#     global _proximity_queue
#     _proximity_queue = WorkerQueue()
#     _proximity_queue.start(_compute_proximity, embedding_model, cluster_centers_tensor)

# def init_diversity_worker(embedding_model, cluster_centers_tensor):
#     global _diversity_queue
#     _diversity_queue = WorkerQueue()
#     _diversity_queue.start(_compute_diversity, embedding_model, cluster_centers_tensor)

# def init_answer_variance_worker(language_model, sampling_params, embedding_model):
#     global _answer_variance_queue
#     _answer_variance_queue = WorkerQueue()
#     _answer_variance_queue.start(_compute_answer_variance, language_model, sampling_params, embedding_model)


# --- public API (called from all.py routes) ---

def mcot(req):
    if _mcot_queue is None:
        raise RuntimeError("mcot worker not initialized — call /start_service first")
    return _mcot_queue.submit(req)

# def confidence(req):
#     if _confidence_queue is None:
#         raise RuntimeError("confidence worker not initialized — call /start_service first")
#     return _confidence_queue.submit(req)

# def gradient(req):
#     if _gradient_queue is None:
#         raise RuntimeError("gradient worker not initialized — call /start_service first")
#     return _gradient_queue.submit(req)

# def proximity(req):
#     if _proximity_queue is None:
#         raise RuntimeError("proximity worker not initialized — call /start_service first")
#     return _proximity_queue.submit(req)

# def diversity(req):
#     if _diversity_queue is None:
#         raise RuntimeError("diversity worker not initialized — call /start_service first")
#     return _diversity_queue.submit(req)

# def answer_variance(req):
#     if _answer_variance_queue is None:
#         raise RuntimeError("answer_variance worker not initialized — call /start_service first")
#     return _answer_variance_queue.submit(req)
