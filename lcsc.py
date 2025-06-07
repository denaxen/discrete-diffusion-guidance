"""Evolutionary search implementation for LCSC checkpoint merging.

This module can be imported by main.py as `import lcsc`.
The public API is:
    run_lcsc(ckpt_paths, config, tokenizer) -> (best_alpha, best_score)
    combine_checkpoints_diff(ckpt_paths, alphas) -> OrderedDict state_dict

All functions are Torch-only and require no Lightning context.
"""

import random
import math
from typing import List, Tuple
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from functools import lru_cache

import diffusion
import dataloader
import eval_utils
import utils

# ----------------------------------------
#  Core helpers
# ----------------------------------------

@lru_cache(maxsize=None)
def _load_state(path: str):
    checkpoint = torch.load(path, map_location='cpu')
    # Lightning checkpoints have the model state dict under 'state_dict' key
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    else:
        # If it's already a state dict (not a Lightning checkpoint)
        return checkpoint

def combine_checkpoints_diff(paths, alphas):
    """θ̂ = Σ αᵢ θᵢ  (all tensors on CPU)."""
    assert len(paths) == len(alphas), "len(paths) != len(alphas)"
    states = [_load_state(p) for p in paths]
    base = states[0]
    mixed = OrderedDict()
    for k in base:
        if isinstance(base[k], torch.Tensor):
            diff_sum = sum(a*(s[k]-base[k]) for a,s in zip(alphas[1:], states[1:]))
            mixed[k] = base[k] + diff_sum
        else:
            mixed[k] = base[k]
    return mixed


# ----------------------------------------
#  Fitness function
# ----------------------------------------

def _fitness(alphas: List[float], ckpt_paths: List[str], config, tokenizer, cache):
    """Compute validation metric for given alphas.

    cache: dict mapping tuple(alphas) -> score to avoid recomputation.
    """
    key = tuple([round(a, 6) for a in alphas])
    if key in cache:
        return cache[key]

    state_dict = combine_checkpoints_diff(ckpt_paths, alphas)

    # Build model and load mixed state
    model = diffusion.Diffusion(config, tokenizer=tokenizer).to('cuda')
    
    # Handle the limiting_distribution buffer that's not in combined checkpoints
    # but is required by the model (it gets recreated during normal Lightning loading)
    if hasattr(model, 'limiting_distribution') and model.limiting_distribution is not None:
        state_dict['limiting_distribution'] = model.limiting_distribution
    
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Choose metric
    with torch.inference_mode():
        if config.lcsc.metric == 'ppl':
            _, val_ds = dataloader.get_dataloaders(config, tokenizer, skip_train=True, valid_seed=config.seed, force_val=True)
            metric_value = eval_utils.compute_ppl(model, val_ds)
        elif config.lcsc.metric == 'gen_ppl':
            samples = []
            for _ in range(config.lcsc.num_sample_batches):
                sample = model.sample()
                samples.extend(model.tokenizer.batch_decode(sample))
            metric_value = eval_utils.compute_generative_ppl(samples, eval_model_name_or_path=config.eval.generative_ppl_model_name_or_path,
                                                            gen_ppl_eval_batch_size=8, max_length=config.model.length)
        elif config.lcsc.metric == 'entropy':
            samples = []
            for _ in range(config.lcsc.num_sample_batches):
                sample = model.sample()
                samples.extend(model.tokenizer.batch_decode(sample))
            tokens = tokenizer.batch_encode_plus(samples, return_tensors='pt', add_special_tokens=False,
                                                max_length=config.model.length, padding='max_length', truncation=True)['input_ids']
            _, counts = torch.unique(tokens, return_counts=True, sorted=False)
            metric_value = -torch.special.entr(counts.float() / counts.sum()).sum().item()
        else:
            raise ValueError(f"Unknown LCSC metric {config.lcsc.metric}")

    cache[key] = metric_value
    torch.cuda.empty_cache()
    return metric_value


def _init_population(K: int, pop_size: int) -> List[List[float]]:
    """Initialize population with diversified EMA-like alphas."""
    population = []
    ema_rates = [0.9, 0.95, 0.97, 0.99, 0.995]
    for g in ema_rates:
        alpha = [g ** (K - 1 - i) for i in range(K)]
        s = sum(alpha)
        population.append([a / s for a in alpha])
    while len(population) < pop_size:
        # random dirichlet (+/- noise)
        alpha = torch.randn(K).tolist()
        s = sum(alpha)
        population.append([a / s for a in alpha])
    return population[:pop_size]


def run_lcsc(*, ckpt_paths: List[str], config, tokenizer) -> Tuple[List[float], float]:
    """Main entry for LCSC evolutionary search.

    Returns best_alphas, best_score (lower is better).
    """
    logger = utils.get_logger('LCSC')

    K = len(ckpt_paths)
    POP = config.lcsc.population_size
    TOP = config.lcsc.top_k
    ITER = config.lcsc.iterations
    MUT_SIGMA = config.lcsc.mutation_sigma
    OFFSPRING = config.lcsc.offspring_per_iter

    population = _init_population(K, POP)
    cache = {}

    # Evaluate initial population
    scored_pop = [(alpha, _fitness(alpha, ckpt_paths, config, tokenizer, cache)) for alpha in population]

    for it in range(ITER):
        scored_pop.sort(key=lambda x: x[1])  # lower better
        parents = [a for a, _ in scored_pop[:TOP]]
        best_score = scored_pop[0][1]
        logger.info(f'Iter {it:03d}: best={best_score:.4f}')

        # Generate offspring
        offspring = []
        for _ in range(OFFSPRING):
            p1, p2 = random.sample(parents, 2)
            child = [(x if random.random() < 0.5 else y) for x, y in zip(p1, p2)]
            # mutation
            child = [a + random.gauss(0, MUT_SIGMA) for a in child]
            # renormalize to sum 1
            s = sum(child)
            child = [a / s for a in child]
            offspring.append(child)

        # Evaluate offspring
        scored_off = [(alpha, _fitness(alpha, ckpt_paths, config, tokenizer, cache)) for alpha in offspring]

        scored_pop.extend(scored_off)
        # keep best POP individuals
        scored_pop.sort(key=lambda x: x[1])
        scored_pop = scored_pop[:POP]

    scored_pop.sort(key=lambda x: x[1])
    best_alpha, best_score = scored_pop[0]
    logger.info(f'Finished search: best score {best_score:.4f}')
    return best_alpha, best_score
