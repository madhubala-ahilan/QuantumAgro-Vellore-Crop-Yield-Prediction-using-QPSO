"""
bayesian_tuner.py  —  QuantumAgro v2  (warm-start restored)
-------------------------------------------------------------
Bayesian Optimisation (via scikit-optimize / gp_minimize) to automatically
find the best *hyperparameters* for each metaheuristic before the main run.

Why Bayesian Optimisation for hyperparameter tuning?
------------------------------------------------------
GA, PSO, and QPSO each have their own hyperparameters (population size,
number of iterations, mutation rate, beta, etc.).  Choosing these manually
or by grid search is expensive.  BO maintains a Gaussian Process surrogate
model of the objective (validation accuracy) and uses an acquisition function
(Expected Improvement) to select the next hyperparameter configuration to
try — using far fewer evaluations than grid/random search.

This adds a genuine second layer of optimisation:
  Layer 1 (BO): find best hyperparameters of the metaheuristic
  Layer 2 (GA / PSO / QPSO): find best FuzzyParams using those hyperparameters

For QPSO specifically, tune_qpso accepts an optional warm_start_vector so
that the BO evaluation runs of QPSO also benefit from the GA warm start —
giving the most accurate picture of QPSO's peak performance.

Reference: Snoek, Larochelle & Adams (2012) "Practical Bayesian Optimization
           of Machine Learning Algorithms", NeurIPS.
"""

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from fuzzy_engine import FuzzyParams, QuantumAgroFIS, evaluate_fis
from genetic_algorithm import GeneticAlgorithm
from pso import PSO
from qpso import QPSO


# ── Quick fitness helpers: run each optimizer once, return best score ─────────

def _quick_ga(vellore_df, pop_size, generations, mutation_rate, crossover_rate):
    ga = GeneticAlgorithm(
        vellore_df=vellore_df,
        pop_size=int(pop_size),
        generations=int(generations),
        mutation_rate=float(mutation_rate),
        crossover_rate=float(crossover_rate),
        seed=42,
        verbose=False,
    )
    _, score, _ = ga.run()
    return score


def _quick_pso(vellore_df, n_particles, max_iter, w_start, c1, c2):
    pso = PSO(
        vellore_df=vellore_df,
        n_particles=int(n_particles),
        max_iter=int(max_iter),
        w_start=float(w_start),
        c1=float(c1),
        c2=float(c2),
        seed=42,
        verbose=False,
    )
    _, score, _ = pso.run()
    return score


def _quick_qpso(vellore_df, n_particles, max_iter, beta, warm_start_vector=None):
    """
    Run QPSO once with the given hyperparameters.
    If warm_start_vector is provided (e.g. GA's best vector), particle-0
    is seeded from it — this is important so BO evaluates QPSO in the
    same mode that the final multi-seed run will use.
    """
    qpso = QPSO(
        vellore_df=vellore_df,
        n_particles=int(n_particles),
        max_iter=int(max_iter),
        beta=float(beta),
        warm_start_vector=warm_start_vector,
        seed=42,
        verbose=False,
    )
    _, score, _ = qpso.run()
    return score


# ── Public tuner functions ────────────────────────────────────────────────────

def tune_ga(vellore_df, n_calls=15, verbose=True):
    """
    Use BO to find best GA hyperparameters.
    Search space: pop_size, generations, mutation_rate, crossover_rate.
    Returns dict of best hyperparameters.
    """
    space = [
        Integer(15, 50,   name='pop_size'),
        Integer(20, 60,   name='generations'),
        Real(0.05, 0.30,  name='mutation_rate'),
        Real(0.60, 0.95,  name='crossover_rate'),
    ]

    @use_named_args(space)
    def objective(**params):
        score = _quick_ga(vellore_df, **params)
        return -score   # minimise negative accuracy

    if verbose:
        print(f"  [BO] Tuning GA hyperparameters ({n_calls} evaluations)...")

    # skopt requires n_calls >= n_initial_points (default 10) + 1
    n_calls = max(n_calls, 11)
    result = gp_minimize(objective, space, n_calls=n_calls,
                         random_state=42, noise=1e-10, verbose=False)

    best = {
        'pop_size':       int(result.x[0]),
        'generations':    int(result.x[1]),
        'mutation_rate':  float(result.x[2]),
        'crossover_rate': float(result.x[3]),
    }
    if verbose:
        print(f"  [BO] Best GA params : {best}  → acc={-result.fun:.4f}")
    return best


def tune_pso(vellore_df, n_calls=15, verbose=True):
    """
    Use BO to find best PSO hyperparameters.
    Search space: n_particles, max_iter, w_start, c1, c2.
    Returns dict of best hyperparameters.
    """
    space = [
        Integer(20, 60,   name='n_particles'),
        Integer(20, 60,   name='max_iter'),
        Real(0.40, 0.95,  name='w_start'),
        Real(1.00, 3.00,  name='c1'),
        Real(1.00, 3.00,  name='c2'),
    ]

    @use_named_args(space)
    def objective(**params):
        score = _quick_pso(vellore_df, **params)
        return -score

    if verbose:
        print(f"  [BO] Tuning PSO hyperparameters ({n_calls} evaluations)...")

    n_calls = max(n_calls, 11)
    result = gp_minimize(objective, space, n_calls=n_calls,
                         random_state=42, noise=1e-10, verbose=False)

    best = {
        'n_particles': int(result.x[0]),
        'max_iter':    int(result.x[1]),
        'w_start':     float(result.x[2]),
        'c1':          float(result.x[3]),
        'c2':          float(result.x[4]),
    }
    if verbose:
        print(f"  [BO] Best PSO params: {best}  → acc={-result.fun:.4f}")
    return best


def tune_qpso(vellore_df, n_calls=15, warm_start_vector=None, verbose=True):
    """
    Use BO to find best QPSO hyperparameters.
    Search space: n_particles, max_iter, beta.
    Returns dict of best hyperparameters.

    Parameters
    ----------
    warm_start_vector : numpy array (length 27), optional.
        If provided, every BO evaluation of QPSO uses this as particle-0's
        initial position — exactly mirroring how the final multi-seed run
        will be configured.  Pass ga_params.as_vector() from the GA step.
    """
    space = [
        Integer(25, 60,   name='n_particles'),
        Integer(30, 70,   name='max_iter'),
        Real(0.60, 1.20,  name='beta'),
    ]

    @use_named_args(space)
    def objective(**params):
        score = _quick_qpso(vellore_df,
                            warm_start_vector=warm_start_vector,
                            **params)
        return -score

    if verbose:
        ws_msg = " (warm-started from GA best)" if warm_start_vector is not None else ""
        print(f"  [BO] Tuning QPSO hyperparameters ({n_calls} evaluations){ws_msg}...")

    n_calls = max(n_calls, 11)
    result = gp_minimize(objective, space, n_calls=n_calls,
                         random_state=42, noise=1e-10, verbose=False)

    best = {
        'n_particles': int(result.x[0]),
        'max_iter':    int(result.x[1]),
        'beta':        float(result.x[2]),
    }
    if verbose:
        print(f"  [BO] Best QPSO params: {best}  → acc={-result.fun:.4f}")
    return best