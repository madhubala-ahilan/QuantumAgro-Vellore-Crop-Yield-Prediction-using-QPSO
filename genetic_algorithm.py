"""
genetic_algorithm.py  —  QuantumAgro v2
----------------------------------------
Real-valued Genetic Algorithm to optimise the 27-dimensional FuzzyParams vector.
Fitness = top-5 hit rate on Vellore eval set.

Operators
---------
Selection  : Tournament (k=3)
Crossover  : BLX-α (alpha=0.5)  — generates offspring outside parent range
Mutation   : Gaussian perturbation (σ = 10% of parameter range)
Elitism    : Best individual carried to next generation unchanged
"""

import numpy as np
from fuzzy_engine import FuzzyParams, QuantumAgroFIS, evaluate_fis


class GeneticAlgorithm:
    def __init__(self,
                 vellore_df,
                 pop_size=30,
                 generations=40,
                 crossover_rate=0.8,
                 mutation_rate=0.15,
                 tournament_k=3,
                 seed=None,
                 verbose=True):

        self.df         = vellore_df
        self.pop_size   = pop_size
        self.generations= generations
        self.cr         = crossover_rate
        self.mr         = mutation_rate
        self.k          = tournament_k
        self.seed       = seed
        self.verbose    = verbose

        self.bounds     = np.array(FuzzyParams.bounds())
        self.dim        = len(self.bounds)

        self.best_params = None
        self.best_score  = -np.inf
        self.history     = []

    # ── fitness ──────────────────────────────────────────────────

    def _fitness(self, chrom):
        params = FuzzyParams(chrom)
        fis    = QuantumAgroFIS(params)
        acc, _, _ = evaluate_fis(fis, self.df)
        return acc

    def _eval_pop(self, pop):
        return np.array([self._fitness(ind) for ind in pop])

    # ── operators ────────────────────────────────────────────────

    def _init_pop(self):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return np.random.uniform(lo, hi, size=(self.pop_size, self.dim))

    def _tournament(self, pop, fits):
        idx  = np.random.choice(len(pop), self.k, replace=False)
        best = idx[np.argmax(fits[idx])]
        return pop[best].copy()

    def _crossover(self, p1, p2, alpha=0.5):
        if np.random.rand() > self.cr:
            return p1.copy(), p2.copy()
        c1, c2 = np.empty(self.dim), np.empty(self.dim)
        for i in range(self.dim):
            lo_p, hi_p = min(p1[i], p2[i]), max(p1[i], p2[i])
            rng = hi_p - lo_p
            c1[i] = np.clip(np.random.uniform(lo_p - alpha*rng, hi_p + alpha*rng),
                            self.bounds[i, 0], self.bounds[i, 1])
            c2[i] = np.clip(np.random.uniform(lo_p - alpha*rng, hi_p + alpha*rng),
                            self.bounds[i, 0], self.bounds[i, 1])
        return c1, c2

    def _mutate(self, ind):
        child = ind.copy()
        for i in range(self.dim):
            if np.random.rand() < self.mr:
                rng = self.bounds[i, 1] - self.bounds[i, 0]
                child[i] += np.random.normal(0, rng * 0.10)
                child[i]  = np.clip(child[i], self.bounds[i, 0], self.bounds[i, 1])
        return child

    # ── main loop ────────────────────────────────────────────────

    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        pop  = self._init_pop()
        fits = self._eval_pop(pop)
        bi   = np.argmax(fits)
        self.best_params = pop[bi].copy()
        self.best_score  = fits[bi]

        for gen in range(self.generations):
            new_pop = [self.best_params.copy()]   # elitism

            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                c1, c2 = self._crossover(p1, p2)
                new_pop.extend([self._mutate(c1), self._mutate(c2)])

            pop  = np.array(new_pop[:self.pop_size])
            fits = self._eval_pop(pop)
            bi   = np.argmax(fits)
            if fits[bi] > self.best_score:
                self.best_score  = fits[bi]
                self.best_params = pop[bi].copy()

            self.history.append(self.best_score)

            if self.verbose and (gen + 1) % 10 == 0:
                print(f"  GA  Gen {gen+1:3d}/{self.generations} | "
                      f"Best: {self.best_score:.4f}")

        return FuzzyParams(self.best_params), self.best_score, self.history
