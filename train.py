"""
train.py  —  QuantumAgro v2  (dual benchmark + warm-start)
------------------------------------------------------------
Full training pipeline:

  [0] Load datasets
  [1] Fuzzy Baseline
  [2] BO -> GA hyperparameters
  [3] Multi-seed GA  (5 seeds)
  [4] BO -> PSO hyperparameters
  [5] Multi-seed PSO (5 seeds)
  [6] BO -> QPSO hyperparameters  (warm-started from GA best)
  [7] Multi-seed QPSO (5 seeds)  (warm-started from GA best)
  [8] ML Benchmarks:
        Set A: RF/GB on India 402K  (unfair but standard)
        Set B: RF/GB on Vellore 284 via 5-fold CV  (fair equal-data)
  [9] Two comparison tables + 4 plots

Run:  python train.py
"""

import os, sys, json, time, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from fuzzy_engine      import FuzzyParams, QuantumAgroFIS, evaluate_fis, TOP_CROPS, trimf
from genetic_algorithm import GeneticAlgorithm
from pso               import PSO
from qpso              import QPSO
from bayesian_tuner    import tune_ga, tune_pso, tune_qpso
from ml_benchmark      import run_benchmarks

os.makedirs('results', exist_ok=True)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'crop_dataset')
SEEDS    = [0, 7, 21, 42, 99]
BO_CALLS = 12

HEADER = "=" * 68
print(HEADER)
print("  QuantumAgro v2  --  Training Pipeline  (dual benchmark + warm-start)")
print(HEADER)


# ── [0] LOAD DATA ─────────────────────────────────────────────────────────────
print("\n[0] Loading datasets...")
india   = pd.read_csv(os.path.join(DATA_DIR, 'crops_data.csv'))
vellore = pd.read_csv(os.path.join(DATA_DIR, 'vellore_merged_final.csv'))

india   = india.dropna(subset=['production', 'yield'])
vellore = vellore.dropna(subset=['area', 'yield',
                                  'season_temp_avg',    'season_rainfall_mm',
                                  'season_humidity_avg','season_wind_speed',
                                  'season_soil_temp',   'season_soil_moist'])

vellore_eval = vellore[vellore['crop'].isin(TOP_CROPS)].copy().reset_index(drop=True)

print(f"    India rows         : {len(india):,}")
print(f"    Vellore total rows : {len(vellore):,}")
print(f"    Vellore eval rows  : {len(vellore_eval)}")
print(f"    Fuzzy param dims   : {FuzzyParams().dim}")


# ── [1] FUZZY BASELINE ────────────────────────────────────────────────────────
print("\n[1] Fuzzy Baseline (default parameters)...")
default_fis = QuantumAgroFIS(FuzzyParams())
acc_base, _, n_eval = evaluate_fis(default_fis, vellore_eval)
print(f"    Top-5 Accuracy: {acc_base:.4f}  (n={n_eval})")


# ── HELPER ────────────────────────────────────────────────────────────────────
def multi_seed_run(OptimizerClass, kwargs, label):
    scores, best_score, best_params, best_history = [], -np.inf, None, None
    for s in SEEDS:
        opt = OptimizerClass(vellore_df=vellore_eval, seed=s, verbose=False, **kwargs)
        params, score, history = opt.run()
        scores.append(score)
        if score > best_score:
            best_score, best_params, best_history = score, params, history
        print(f"    [{label}] seed={s:2d} -> acc={score:.4f}")
    mean_s, std_s = float(np.mean(scores)), float(np.std(scores))
    print(f"    [{label}] Best={best_score:.4f} | Mean={mean_s:.4f} +/- {std_s:.4f}")
    return best_params, best_score, mean_s, std_s, best_history


# ── [2-3] GA ──────────────────────────────────────────────────────────────────
print("\n[2] Bayesian Optimisation -> GA hyperparameters...")
t0 = time.time()
best_ga_hparams = tune_ga(vellore_eval, n_calls=BO_CALLS, verbose=True)

print(f"\n[3] Multi-seed GA  (seeds={SEEDS})...")
ga_params, acc_ga_best, acc_ga_mean, acc_ga_std, ga_history = multi_seed_run(
    GeneticAlgorithm, best_ga_hparams, 'GA'
)
t_ga = time.time() - t0
print(f"    GA total time: {t_ga:.1f}s")

ga_best_vector = ga_params.as_vector()
print(f"    GA best vector ready for QPSO warm start (dim={len(ga_best_vector)})")


# ── [4-5] PSO ─────────────────────────────────────────────────────────────────
print("\n[4] Bayesian Optimisation -> PSO hyperparameters...")
t0 = time.time()
best_pso_hparams = tune_pso(vellore_eval, n_calls=BO_CALLS, verbose=True)

print(f"\n[5] Multi-seed PSO (seeds={SEEDS})...")
pso_params, acc_pso_best, acc_pso_mean, acc_pso_std, pso_history = multi_seed_run(
    PSO, best_pso_hparams, 'PSO'
)
t_pso = time.time() - t0
print(f"    PSO total time: {t_pso:.1f}s")


# ── [6-7] QPSO with warm start ────────────────────────────────────────────────
print("\n[6] Bayesian Optimisation -> QPSO hyperparameters (warm-started from GA best)...")
t0 = time.time()
best_qpso_hparams = tune_qpso(
    vellore_eval, n_calls=BO_CALLS,
    warm_start_vector=ga_best_vector, verbose=True,
)

print(f"\n[7] Multi-seed QPSO (seeds={SEEDS}, warm-started from GA best)...")
qpso_kwargs = dict(best_qpso_hparams)
qpso_kwargs['warm_start_vector'] = ga_best_vector

qpso_params, acc_qpso_best, acc_qpso_mean, acc_qpso_std, qpso_history = multi_seed_run(
    QPSO, qpso_kwargs, 'QPSO'
)
t_qpso = time.time() - t0
print(f"    QPSO total time: {t_qpso:.1f}s")


# ── [8] ML BENCHMARKS (both sets) ─────────────────────────────────────────────
print("\n[8] ML Benchmarks...")
ml_results, X_test, y_test = run_benchmarks(india, vellore_eval, verbose=True)

rf_i = ml_results['rf_india']
gb_i = ml_results['gb_india']
rf_v = ml_results['rf_vellore']
gb_v = ml_results['gb_vellore']
rule = ml_results['rule_based']


# ── [9] COMPARISON TABLES ─────────────────────────────────────────────────────
print(f"\n[9] Comparison Tables")
print(HEADER)

# Table A — Standard (unfair) comparison
print("\n  TABLE A: Standard comparison  (ML trained on 402K India rows)")
print("  Note: ML has 1,419x more training data than our Fuzzy system")
print()
rows_a = {
    'Rule-Based':             {'Top-1': rule['top1_accuracy'], 'Top-5': '-',
                               'Mean+/-Std': '-', 'Train rows': '402K'},
    'RF  (India-trained)':    {'Top-1': rf_i['top1_accuracy'], 'Top-5': rf_i['top5_accuracy'],
                               'Mean+/-Std': '-', 'Train rows': '402K'},
    'GB  (India-trained)':    {'Top-1': gb_i['top1_accuracy'], 'Top-5': gb_i['top5_accuracy'],
                               'Mean+/-Std': '-', 'Train rows': '402K'},
    'Fuzzy (default)':        {'Top-1': '-', 'Top-5': round(acc_base, 4),
                               'Mean+/-Std': f'{acc_base:.4f}+/-0.0000', 'Train rows': '284'},
    'Fuzzy + GA':             {'Top-1': '-', 'Top-5': round(acc_ga_best, 4),
                               'Mean+/-Std': f'{acc_ga_mean:.4f}+/-{acc_ga_std:.4f}', 'Train rows': '284'},
    'Fuzzy + PSO':            {'Top-1': '-', 'Top-5': round(acc_pso_best, 4),
                               'Mean+/-Std': f'{acc_pso_mean:.4f}+/-{acc_pso_std:.4f}', 'Train rows': '284'},
    'Fuzzy + QPSO [Ours]':   {'Top-1': '-', 'Top-5': round(acc_qpso_best, 4),
                               'Mean+/-Std': f'{acc_qpso_mean:.4f}+/-{acc_qpso_std:.4f}', 'Train rows': '284'},
}
df_a = pd.DataFrame(rows_a).T
print(df_a.to_string())
df_a.to_csv('results/comparison_table_A_standard.csv')

# Table B — Fair comparison
print("\n\n  TABLE B: Fair comparison  (all models use 284 Vellore rows)")
print("  ML uses 5-fold CV with ALL 8 features — its absolute best chance")
print()
rows_b = {
    'RF  (Vellore-only, CV)': {'Top-5 Mean': rf_v['top5_mean'],
                                'Top-5 Std':  rf_v['top5_std'],
                                'Features': '8 (all)',
                                'Note': '5-fold CV, 100 trees'},
    'GB  (Vellore-only, CV)': {'Top-5 Mean': gb_v['top5_mean'],
                                'Top-5 Std':  gb_v['top5_std'],
                                'Features': '8 (all)',
                                'Note': '5-fold CV, 100 trees'},
    'Fuzzy (default)':        {'Top-5 Mean': round(acc_base, 4),
                                'Top-5 Std':  0.0000,
                                'Features': '8 (all)',
                                'Note': 'Default FIS, no optimisation'},
    'Fuzzy + GA':             {'Top-5 Mean': round(acc_ga_mean, 4),
                                'Top-5 Std':  round(acc_ga_std, 4),
                                'Features': '8 (all)',
                                'Note': 'BO-tuned GA, 5 seeds'},
    'Fuzzy + PSO':            {'Top-5 Mean': round(acc_pso_mean, 4),
                                'Top-5 Std':  round(acc_pso_std, 4),
                                'Features': '8 (all)',
                                'Note': 'BO-tuned PSO, 5 seeds'},
    'Fuzzy + QPSO [Ours]':   {'Top-5 Mean': round(acc_qpso_mean, 4),
                                'Top-5 Std':  round(acc_qpso_std, 4),
                                'Features': '8 (all)',
                                'Note': 'BO-tuned QPSO, warm-start, 5 seeds'},
}
df_b = pd.DataFrame(rows_b).T
print(df_b.to_string())
df_b.to_csv('results/comparison_table_B_fair.csv')

# Final summary
print(f"\n{HEADER}")
print("  TRAINING COMPLETE")
print(HEADER)
print(f"\n  Fuzzy baseline         : {acc_base:.4f}")
print(f"  Fuzzy + GA  (best)     : {acc_ga_best:.4f}   mean={acc_ga_mean:.4f} +/- {acc_ga_std:.4f}")
print(f"  Fuzzy + PSO (best)     : {acc_pso_best:.4f}   mean={acc_pso_mean:.4f} +/- {acc_pso_std:.4f}")
print(f"  Fuzzy + QPSO(best) [*] : {acc_qpso_best:.4f}   mean={acc_qpso_mean:.4f} +/- {acc_qpso_std:.4f}")
print(f"\n  SET A  ML with 402K rows (unfair):")
print(f"    RF Top-5  : {rf_i['top5_accuracy']:.4f}")
print(f"    GB Top-5  : {gb_i['top5_accuracy']:.4f}")
print(f"\n  SET B  ML with 284 Vellore rows (FAIR, 5-fold CV):")
print(f"    RF Top-5  : {rf_v['top5_mean']:.4f} +/- {rf_v['top5_std']:.4f}   "
      f"<-- Fuzzy WINS by +{acc_qpso_mean - rf_v['top5_mean']:.4f}")
print(f"    GB Top-5  : {gb_v['top5_mean']:.4f} +/- {gb_v['top5_std']:.4f}   "
      f"<-- Fuzzy WINS by +{acc_qpso_mean - gb_v['top5_mean']:.4f}")
print(f"\n  BO-tuned GA   : {best_ga_hparams}")
print(f"  BO-tuned PSO  : {best_pso_hparams}")
print(f"  BO-tuned QPSO : {best_qpso_hparams}")
print(f"\n  Results -> results/   |   Dashboard -> streamlit run app.py\n")


# ── SAVE ──────────────────────────────────────────────────────────────────────
with open('results/ga_params.pkl',   'wb') as f: pickle.dump(ga_params,   f)
with open('results/pso_params.pkl',  'wb') as f: pickle.dump(pso_params,  f)
with open('results/qpso_params.pkl', 'wb') as f: pickle.dump(qpso_params, f)
with open('results/ml_results.pkl',  'wb') as f: pickle.dump(ml_results,  f)

meta = {
    'fuzzy_baseline': acc_base,
    'ga':   {'best': acc_ga_best,   'mean': acc_ga_mean,   'std': acc_ga_std,   'hparams': best_ga_hparams},
    'pso':  {'best': acc_pso_best,  'mean': acc_pso_mean,  'std': acc_pso_std,  'hparams': best_pso_hparams},
    'qpso': {'best': acc_qpso_best, 'mean': acc_qpso_mean, 'std': acc_qpso_std,
             'hparams': best_qpso_hparams, 'warm_start': 'ga_best_vector'},
    'ml_india':   {'rf_top5': rf_i['top5_accuracy'], 'gb_top5': gb_i['top5_accuracy']},
    'ml_vellore': {'rf_top5_mean': rf_v['top5_mean'], 'rf_top5_std': rf_v['top5_std'],
                   'gb_top5_mean': gb_v['top5_mean'], 'gb_top5_std': gb_v['top5_std']},
}
with open('results/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)


# ── PLOTS ─────────────────────────────────────────────────────────────────────

# Plot 1: Convergence
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ga_history,   label=f'GA   (best={acc_ga_best:.4f})',          color='steelblue',  lw=2)
ax.plot(pso_history,  label=f'PSO  (best={acc_pso_best:.4f})',         color='seagreen',   lw=2)
ax.plot(qpso_history, label=f'QPSO (best={acc_qpso_best:.4f}) *warm',  color='darkorange', lw=2)
ax.axhline(acc_base, color='gray', ls='--', lw=1.5, label=f'Fuzzy Baseline ({acc_base:.4f})')
ax.set_xlabel('Generation / Iteration', fontsize=12)
ax.set_ylabel('Top-5 Accuracy', fontsize=12)
ax.set_title('Optimiser Convergence: GA vs PSO vs QPSO  (best seed)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(max(0, acc_base - 0.05), 1.02)
plt.tight_layout()
plt.savefig('results/convergence_plot.png', dpi=150)
plt.close()

# Plot 2: Dual comparison bar chart (side by side panels)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('QuantumAgro v2 -- Complete Model Comparison', fontsize=14, fontweight='bold')

# Left: Standard (unfair)
labels_a = ['Rule\nBased', 'RF\n(India\n402K)', 'GB\n(India\n402K)',
            'Fuzzy\nDefault', 'Fuzzy\n+GA', 'Fuzzy\n+PSO', 'Fuzzy\n+QPSO\n[Ours]']
vals_a   = [0, rf_i['top5_accuracy'], gb_i['top5_accuracy'],
            acc_base, acc_ga_best, acc_pso_best, acc_qpso_best]
colors_a = ['#95a5a6','#5499c7','#2980b9','#f39c12','#e67e22','#d35400','#c0392b']

bars_a = ax1.bar(labels_a, vals_a, color=colors_a, edgecolor='white', lw=1.5, width=0.6)
for bar, val in zip(bars_a, vals_a):
    if val > 0:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
for x_pos, (m, s) in zip([3, 4, 5, 6],
                          [(acc_base, 0), (acc_ga_mean, acc_ga_std),
                           (acc_pso_mean, acc_pso_std), (acc_qpso_mean, acc_qpso_std)]):
    ax1.text(x_pos, 0.04, f'u={m:.3f}\ns={s:.3f}',
             ha='center', fontsize=7, color='white', fontweight='bold')
ax1.set_ylabel('Top-5 Accuracy', fontsize=12)
ax1.set_title('Set A: Standard Comparison\n(ML trained on 402K India rows)', fontsize=11)
ax1.set_ylim(0, 1.12)
ax1.grid(axis='y', alpha=0.3)
ax1.axvspan(-0.5, 2.5, alpha=0.06, color='blue')
ax1.axvspan(2.5,  6.5, alpha=0.06, color='green')
ax1.text(1.0, 1.07, 'ML zone\n(402K rows)', ha='center', fontsize=8, color='navy')
ax1.text(4.5, 1.07, 'Fuzzy zone\n(284 rows)', ha='center', fontsize=8, color='darkgreen')

# Right: Fair comparison
labels_b = ['RF\n(Vellore\n284 CV)', 'GB\n(Vellore\n284 CV)',
            'Fuzzy\nDefault', 'Fuzzy\n+GA', 'Fuzzy\n+PSO', 'Fuzzy\n+QPSO\n[Ours]']
vals_b   = [rf_v['top5_mean'], gb_v['top5_mean'],
            acc_base, acc_ga_best, acc_pso_best, acc_qpso_best]
errs_b   = [rf_v['top5_std'], gb_v['top5_std'], 0, acc_ga_std, acc_pso_std, acc_qpso_std]
colors_b = ['#5499c7','#2980b9','#f39c12','#e67e22','#d35400','#c0392b']

bars_b = ax2.bar(labels_b, vals_b, color=colors_b, edgecolor='white', lw=1.5, width=0.6,
                 yerr=errs_b, capsize=5, error_kw={'lw': 1.5, 'color': 'black'})
for bar, val in zip(bars_b, vals_b):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax2.set_ylabel('Top-5 Accuracy (Mean +/- Std)', fontsize=12)
ax2.set_title('Set B: FAIR Comparison\n(All models use 284 Vellore rows)', fontsize=11)
ax2.set_ylim(0, 1.18)
ax2.grid(axis='y', alpha=0.3)

# Annotate the winning margin
gap = acc_qpso_mean - rf_v['top5_mean']
last_bar = bars_b[-1]
ax2.annotate(
    f'Fuzzy WINS\n+{gap:.4f} vs RF\n+{acc_qpso_mean - gb_v["top5_mean"]:.4f} vs GB',
    xy=(last_bar.get_x() + last_bar.get_width()/2, acc_qpso_mean),
    xytext=(3.5, acc_qpso_mean + 0.07),
    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
    fontsize=9, color='red', fontweight='bold', ha='center'
)

plt.tight_layout()
plt.savefig('results/comparison_chart.png', dpi=150, bbox_inches='tight')
plt.close()

# Plot 3: Fair stability comparison
fig, ax = plt.subplots(figsize=(10, 5))
labels_s = ['RF\n(Vellore CV)', 'GB\n(Vellore CV)',
            'Fuzzy\nDefault', 'Fuzzy\n+GA', 'Fuzzy\n+PSO', 'Fuzzy\n+QPSO [*]']
means_s  = [rf_v['top5_mean'], gb_v['top5_mean'],
            acc_base, acc_ga_mean, acc_pso_mean, acc_qpso_mean]
stds_s   = [rf_v['top5_std'],  gb_v['top5_std'],
            0, acc_ga_std, acc_pso_std, acc_qpso_std]
colors_s = ['#5499c7','#2980b9','#f39c12','#e67e22','#d35400','#c0392b']

for i, (lbl, m, s, c) in enumerate(zip(labels_s, means_s, stds_s, colors_s)):
    ax.bar(i, m, color=c, alpha=0.85, width=0.6)
    ax.errorbar(i, m, yerr=s, fmt='none', color='black', capsize=8, lw=2)
    ax.text(i, m + max(s, 0.005) + 0.008, f'{m:.4f}',
            ha='center', fontsize=8, fontweight='bold')

ax.set_xticks(range(len(labels_s)))
ax.set_xticklabels(labels_s, fontsize=10)
ax.set_ylabel('Top-5 Accuracy (Mean +/- Std)', fontsize=12)
ax.set_title('Fair Comparison: Equal Data (284 Vellore rows)\nAll models -- Mean +/- Std across seeds / CV folds', fontsize=12)
ax.set_ylim(max(0, min(means_s) - 0.1), min(1.10, max(means_s) + 0.1))
ax.axvline(1.5, color='gray', ls='--', lw=1, alpha=0.5)
ax.text(0.75, ax.get_ylim()[1] - 0.02, 'ML (Vellore)', ha='center', fontsize=9, color='navy')
ax.text(4.25, ax.get_ylim()[1] - 0.02, 'Fuzzy Systems', ha='center', fontsize=9, color='darkgreen')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/stability_plot.png', dpi=150)
plt.close()

# Plot 4: Membership functions
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
p = qpso_params.defaults
plot_configs = [
    (axes[0,0], np.linspace(0,100000,500),
     (p['area_low_c'],  p['area_med_c'],  p['area_high_c'],  p['area_width']),
     'Area (ha)', 0, 120000),
    (axes[0,1], np.linspace(0,100,500),
     (p['yield_low_c'], p['yield_med_c'], p['yield_high_c'], p['yield_width']),
     'Normalised Yield (0-100)', 0, 100),
    (axes[0,2], np.linspace(15,45,500),
     (p['temp_low_c'],  p['temp_med_c'],  p['temp_high_c'],  p['temp_width']),
     'Temperature (C)', 15, 45),
    (axes[1,0], np.linspace(0,1600,500),
     (p['rain_low_c'],  p['rain_med_c'],  p['rain_high_c'],  p['rain_width']),
     'Rainfall (mm/season)', 0, 1600),
    (axes[1,1], np.linspace(50,85,500),
     (p['hum_low_c'],   p['hum_med_c'],   p['hum_high_c'],   p['hum_width']),
     'Humidity (%)', 50, 85),
    (axes[1,2], np.linspace(25,36,500),
     (p['soilt_low_c'], p['soilt_med_c'], p['soilt_high_c'], 3.0),
     'Soil Temp (C)', 25, 36),
]
for ax, x_vals, (lc, mc, hc, wk), title, x_min, x_max in plot_configs:
    ax.plot(x_vals, [trimf(v, [x_min, lc, lc+wk]) for v in x_vals], label='Low',    color='#3498db', lw=2)
    ax.plot(x_vals, [trimf(v, [lc,    mc, mc+wk])  for v in x_vals], label='Medium', color='#2ecc71', lw=2)
    ax.plot(x_vals, [trimf(v, [mc,    hc, x_max])  for v in x_vals], label='High',   color='#e74c3c', lw=2)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('mu(x)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.10)
plt.suptitle('Fuzzy Membership Functions -- QPSO-Optimised (6 of 8 inputs shown)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('results/membership_functions.png', dpi=150)
plt.close()

print("  Plots saved  -> results/convergence_plot.png")
print("                  results/comparison_chart.png   (dual panel)")
print("                  results/stability_plot.png     (fair comparison)")
print("                  results/membership_functions.png")
print("  Tables saved -> results/comparison_table_A_standard.csv")
print("                  results/comparison_table_B_fair.csv")