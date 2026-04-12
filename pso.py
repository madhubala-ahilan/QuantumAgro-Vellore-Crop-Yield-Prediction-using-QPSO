"""
pso.py  —  QuantumAgro v2
--------------------------
Classical Particle Swarm Optimisation (Kennedy & Eberhart, 1995).

In classical PSO each particle has a velocity that is updated using:
  v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)
  x(t+1) = x(t) + v(t+1)

Where:
  w    = inertia weight (decreases linearly → more exploitation over time)
  c1   = cognitive coefficient (attraction to personal best)
  c2   = social coefficient (attraction to global best)
  r1,r2 = random uniform samples

This serves as the intermediate comparison between GA (evolutionary) and
QPSO (quantum-inspired), forming the progression:
  Rule-Based → RF → GB → Fuzzy → GA → PSO → QPSO
"""

import numpy as np
from fuzzy_engine import FuzzyParams, QuantumAgroFIS, evaluate_fis


class PSO:
    def __init__(self,
                 vellore_df,
                 n_particles=30,
                 max_iter=40,
                 w_start=0.9,
                 w_end=0.4,
                 c1=2.0,
                 c2=2.0,
                 seed=None,
                 verbose=True):
        """
        Parameters
        ----------
        w_start / w_end : inertia weight linearly decays from w_start → w_end
        c1              : cognitive (personal best) coefficient
        c2              : social (global best) coefficient
        """
        self.df         = vellore_df
        self.n          = n_particles
        self.max_iter   = max_iter
        self.w_start    = w_start
        self.w_end      = w_end
        self.c1         = c1
        self.c2         = c2
        self.seed       = seed
        self.verbose    = verbose

        self.bounds      = np.array(FuzzyParams.bounds())
        self.dim         = len(self.bounds)
        self.lo          = self.bounds[:, 0]
        self.hi          = self.bounds[:, 1]
        self.v_max       = (self.hi - self.lo) * 0.2   # velocity clamping

        self.best_params = None
        self.best_score  = -np.inf
        self.history     = []

    def _fitness(self, pos):
        params = FuzzyParams(pos)
        fis    = QuantumAgroFIS(params)
        acc, _, _ = evaluate_fis(fis, self.df)
        return acc

    def run(self):
        if self.seed is not None:
            np.random.seed(self.seed)

        # Initialise positions and velocities
        positions  = np.random.uniform(self.lo, self.hi, (self.n, self.dim))
        velocities = np.random.uniform(-(self.hi - self.lo) * 0.1,
                                        (self.hi - self.lo) * 0.1,
                                       (self.n, self.dim))

        pbest      = positions.copy()
        pbest_fit  = np.array([self._fitness(p) for p in positions])

        gi         = np.argmax(pbest_fit)
        gbest      = pbest[gi].copy()
        self.best_score  = pbest_fit[gi]
        self.best_params = gbest.copy()

        for t in range(self.max_iter):
            # Linear inertia decay
            w = self.w_start - (self.w_start - self.w_end) * t / self.max_iter

            r1 = np.random.rand(self.n, self.dim)
            r2 = np.random.rand(self.n, self.dim)

            # Velocity update
            velocities = (w * velocities
                          + self.c1 * r1 * (pbest - positions)
                          + self.c2 * r2 * (gbest - positions))
            # Velocity clamping
            velocities = np.clip(velocities, -self.v_max, self.v_max)

            # Position update
            positions  = np.clip(positions + velocities, self.lo, self.hi)

            # Evaluate & update bests
            for i in range(self.n):
                f = self._fitness(positions[i])
                if f > pbest_fit[i]:
                    pbest_fit[i] = f
                    pbest[i]     = positions[i].copy()
                if f > self.best_score:
                    self.best_score  = f
                    self.best_params = positions[i].copy()
                    gbest            = positions[i].copy()

            self.history.append(self.best_score)

            if self.verbose and (t + 1) % 10 == 0:
                print(f"  PSO Iter {t+1:3d}/{self.max_iter} | "
                      f"Best: {self.best_score:.4f}")

        return FuzzyParams(self.best_params), self.best_score, self.history
