# avalanches/parallel_wrapper.py
import os, math, importlib, numpy as np
from multiprocessing import get_context

__all__ = ["parallel"]

def _init_threads(threads_per_worker):
    # Avoid BLAS oversubscription
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    # Try to limit PyTorch threads if available
    try:
        import torch
        torch.set_num_threads(threads_per_worker)
        torch.set_num_interop_threads(threads_per_worker)
    except Exception:
        pass

def _resolve_func(module_name, func_name):
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    return fn

def _worker_call(args):
    """
    Top-level worker (must be picklable!).
    Re-imports the function by module/name, sets seeds/threads, then calls it.
    """
    (module_name, func_name, split_arg, chunk, seed, threads_per_worker, kwargs) = args

    _init_threads(threads_per_worker)

    # Seeds
    if seed is not None:
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except Exception:
            pass

    # Build per-chunk kwargs
    kw = dict(kwargs)
    kw[split_arg] = int(chunk)
    kw.setdefault("print_every", kwargs.get("print_every", 0))

    func = _resolve_func(module_name, func_name)
    return func(**kw)

def _merge_results(parts):
    first = parts[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(parts, axis=0)
    if isinstance(first, (list, tuple)):
        out = []
        for p in parts: out.extend(list(p))
        return out
    if isinstance(first, dict):
        merged = {}
        for k in first.keys():
            vals = [p[k] for p in parts]
            if isinstance(vals[0], np.ndarray):
                merged[k] = np.concatenate(vals, axis=0)
            elif isinstance(vals[0], (list, tuple)):
                tmp = []
                for v in vals: tmp.extend(list(v))
                merged[k] = tmp
            else:
                merged[k] = vals
        return merged
    return parts

def parallel(func, split_arg="num_samples", num_workers=None,
             base_seed=0, threads_per_worker=1, **kwargs):
    """
    Generic CPU parallelizer. Pass a top-level function and kwargs.
    One kwarg (default 'num_samples') will be divided across workers.
    """
    if not callable(func):
        raise TypeError("First arg must be a function, e.g. parallel(run_simulation, ..., num_samples=...).")
    total = kwargs.get(split_arg)
    if total is None:
        raise ValueError(f"Function must have '{split_arg}' in kwargs to split; got {list(kwargs.keys())}")

    if num_workers is None:
        num_workers = max(1, min(os.cpu_count() or 1, 8))

    # Split evenly
    sizes, remaining = [], total
    for i in range(num_workers):
        chunk = math.ceil(remaining / (num_workers - i))
        if chunk <= 0: break
        sizes.append(chunk)
        remaining -= chunk

    # Pass function by module/name so workers can import it
    module_name = func.__module__
    func_name = func.__name__

    ctx = get_context("spawn")
    with ctx.Pool(processes=len(sizes)) as pool:
        parts = pool.map(
            _worker_call,
            [
                (module_name, func_name, split_arg, s, (base_seed or 0) + i, threads_per_worker, kwargs)
                for i, s in enumerate(sizes)
            ],
        )

    return _merge_results(parts)