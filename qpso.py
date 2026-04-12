"""
qpso.py  —  QuantumAgro v2  (warm-start restored)
--------------------------------------------------
Quantum-behaved Particle Swarm Optimisation (Sun et al., 2004).

Key difference from classical PSO
----------------------------------
In classical PSO the trajectory of each particle is deterministic given
its velocity.  In QPSO the particle is modelled as existing in a quantum
potential well centred on its personal attractor p_i.  The position update
is governed by a quantum delta potential model:

    mbest   = (1/N) * Σ pbest_i          (mean best — global guide)
    phi     ~ U(0,1)
    p_i     = phi * pbest_i + (1-phi) * gbest    (local attractor)
    u       ~ U(0,1)
    x_i(t+1) = p_i  ±  β |mbest - x_i(t)| * ln(1/u)

The  ln(1/u)  term is the quantum tunnelling analogy — it allows particles
to escape local optima that trap classical PSO.  β (contraction-expansion
coefficient) controls the balance between exploration and exploitation.

Warm start
----------
If warm_start_vector is supplied, particle-0 is initialised to that vector
(clipped to bounds) instead of a random position.  In QuantumAgro this is
set to the best parameter vector found by GA, giving QPSO a strong head
start in an already-promising region of the search space.

Reference: Sun, J., Xu, W., & Feng, B. (2004). A global search strategy
           of quantum-behaved particle swarm optimization. Proceedings of
           the IEEE Conference on Cybernetics and Intelligent Systems.
"""

import numpy as np
from fuzzy_engine import FuzzyParams, QuantumAgroFIS, evaluate_fis


class QPSO:
    def __init__(self,
                 vellore_df,
                 n_particles=40,
                 max_iter=50,
                 beta=0.90,
                 seed=None,
                 warm_start_vector=None,
                 verbose=True):
        """
        Parameters
        ----------
        vellore_df        : Vellore evaluation DataFrame
        n_particles       : swarm size
        max_iter          : number of QPSO iterations
        beta              : contraction-expansion coefficient.
                            Higher (>0.8) = more exploration; lower = more exploitation.
                            Decays linearly from beta → beta/2 over iterations.
        seed              : random seed for reproducibility
        warm_start_vector : optional numpy array (length 27).
                            If provided, particle-0 is initialised here instead
                            of a random position.  Pass ga_params.as_vector()
                            from the GA run to hot-start QPSO.
        verbose           : print progress every 10 iterations
        """
        self.df               = vellore_df
        self.n                = n_particles
        self.max_iter         = max_iter
        self.beta             = beta
        self.seed             = seed
        self.warm_start       = (np.asarray(warm_start_vector, dtype=float)
                                 if warm_start_vector is not None else None)
        self.verbose          = verbose

        self.bounds           = np.array(FuzzyParams.bounds())   # (27, 2)
        self.dim              = len(self.bounds)

        self.best_params      = None
        self.best_score       = -np.inf
        self.history          = []

    # ── helpers ──────────────────────────────────────────────────────

    def _fitness(self, pos):
        fis = QuantumAgroFIS(FuzzyParams(pos))
        acc, _, _ = evaluate_fis(fis, self.df)
        return acc

    def _clip(self, pos):
        return np.clip(pos, self.bounds[:, 0], self.bounds[:, 1])

    # ── main loop ────────────────────────────────────────────────────

    def run(self):
        """
        Execute QPSO and return (FuzzyParams, best_score, history).

        history : list of best Top-5 accuracy after each iteration
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        lo, hi    = self.bounds[:, 0], self.bounds[:, 1]
        positions = np.random.uniform(lo, hi, (self.n, self.dim)) ####

        # ── Warm start: inject GA's best vector into particle-0 ──────
        if self.warm_start is not None:
            positions[0] = self._clip(self.warm_start.copy())

        pbest     = positions.copy()
        pbest_fit = np.array([self._fitness(p) for p in positions])

        gi               = np.argmax(pbest_fit)
        gbest            = pbest[gi].copy()
        self.best_score  = pbest_fit[gi]
        self.best_params = gbest.copy()

        for t in range(self.max_iter):
            # Beta decays linearly:  β(t) = β * (1 - 0.5 * t/T)
            beta_t = self.beta * (1.0 - 0.5 * t / self.max_iter)

            # Mean best position across all personal bests (quantum centre)
            mbest = np.mean(pbest, axis=0)

            for i in range(self.n):
                phi = np.random.rand(self.dim)
                p_i = phi * pbest[i] + (1.0 - phi) * gbest       # local attractor

                u    = np.random.uniform(1e-10, 1.0, self.dim)
                sign = np.where(np.random.rand(self.dim) < 0.5, 1.0, -1.0)

                # Quantum position update
                new_pos = p_i + sign * beta_t * np.abs(mbest - positions[i]) * np.log(1.0 / u)
                new_pos = self._clip(new_pos)
                positions[i] = new_pos

                f = self._fitness(new_pos)

                if f > pbest_fit[i]:
                    pbest_fit[i] = f
                    pbest[i]     = new_pos.copy()

                if f > self.best_score:
                    self.best_score  = f
                    self.best_params = new_pos.copy()
                    gbest            = new_pos.copy()

            self.history.append(self.best_score)

            if self.verbose and (t + 1) % 10 == 0:
                ws = " [warm-start]" if (self.warm_start is not None and t == 0) else ""
                print(f"  QPSO Iter {t+1:3d}/{self.max_iter} | "
                      f"Best: {self.best_score:.4f}{ws}")

        return FuzzyParams(self.best_params), self.best_score, self.history