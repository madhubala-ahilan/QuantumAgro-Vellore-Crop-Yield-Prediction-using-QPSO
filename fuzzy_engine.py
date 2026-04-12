"""
fuzzy_engine.py  —  QuantumAgro v2
-----------------------------------
Extended Fuzzy Inference System with 8 input variables:
  Original (3): area, yield_history, temperature
  New     (5): rainfall, humidity, wind_speed, soil_temp, soil_moisture

Parameter vector grows from 12 → 27 dimensions.
All membership functions are triangular (trimf) for interpretability.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────
# MEMBERSHIP FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def trimf(x, params):
    """Triangular MF.  params = [a, b, c]  (left foot, peak, right foot)"""
    a, b, c = params
    left  = (x - a) / (b - a + 1e-9) if b != a else 0.0
    right = (c - x) / (c - b + 1e-9) if c != b else 0.0
    return float(np.clip(min(left, right), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────
# TOP CROPS & DOMAIN KNOWLEDGE
# ─────────────────────────────────────────────────────────────────

TOP_CROPS = [
    'Coconut', 'Sugarcane', 'Rice', 'Banana', 'Groundnut',
    'Ragi',    'Jowar',     'Guar seed', 'Cotton(lint)', 'Maize'
]

# Season affinity: 0=unsuitable, 1=possible, 2=preferred
#                   Coco  Sugg  Rice  Ban   Grnt  Ragi  Jowr  Guar  Cott  Maiz
SEASON_AFFINITY = {
    'Whole Year': [2,    2,    1,    2,    1,    1,    1,    2,    1,    1],
    'Kharif':     [0,    0,    2,    2,    2,    2,    2,    0,    2,    2],
    'Rabi':       [0,    0,    0,    0,    2,    2,    2,    0,    2,    2],
    'Summer':     [0,    0,    2,    0,    1,    0,    0,    0,    0,    0],
    'Autumn':     [0,    0,    2,    0,    0,    0,    0,    0,    0,    0],
    'Winter':     [0,    0,    2,    0,    0,    0,    0,    0,    0,    0],
}

# Average yield per crop in Vellore (tons/ha or nuts/ha for Coconut)
CROP_YIELD_BASE = {
    'Coconut':      11391.29,
    'Sugarcane':      114.77,
    'Rice':             3.81,
    'Banana':          36.39,
    'Groundnut':        2.07,
    'Ragi':             2.76,
    'Jowar':            1.29,
    'Guar seed':        8.00,
    'Cotton(lint)':     2.03,
    'Maize':            4.88,
}

# India-wide 99th percentile yield for normalisation (0-100 scale)
YIELD_NORM_MAX = {
    'Coconut':      43958,
    'Sugarcane':     2247,
    'Rice':           223,
    'Banana':         258,
    'Groundnut':      599,
    'Ragi':            21,
    'Jowar':           22,
    'Guar seed':       40,
    'Cotton(lint)':   300,
    'Maize':         1494,
}

# Crops that benefit from high rainfall / moisture
RAIN_LOVERS    = [0, 1, 2, 3]   # Coconut, Sugarcane, Rice, Banana
DROUGHT_HARDY  = [4, 5, 6, 7]   # Groundnut, Ragi, Jowar, Guar seed
HEAT_LOVERS    = [0, 1, 3]      # Coconut, Sugarcane, Banana


# ─────────────────────────────────────────────────────────────────
# FUZZY PARAMETERS  (27-dimensional vector)
# ─────────────────────────────────────────────────────────────────

class FuzzyParams:
    """
    Holds all 27 MF parameters.  GA / PSO / QPSO optimise this vector.

    Layout (3 params each = low_c, med_c, high_c + 1 shared width):
      [0-3]   area          (4 params)
      [4-7]   yield         (4 params)
      [8-11]  temperature   (4 params)
      [12-14] rainfall      (3 params: low_c, med_c, high_c; width shared)
      [15]    rainfall_w
      [16-18] humidity      (3 params)
      [19]    humidity_w
      [20-22] wind_speed    (3 params)
      [23]    wind_w
      [24-26] soil_temp     (3 params)   ← soil_moist folded into rules
    Total = 27
    """

    _KEYS = [
        # area (ha)
        'area_low_c', 'area_med_c', 'area_high_c', 'area_width',
        # normalised yield (0-100)
        'yield_low_c', 'yield_med_c', 'yield_high_c', 'yield_width',
        # temperature (°C)
        'temp_low_c', 'temp_med_c', 'temp_high_c', 'temp_width',
        # rainfall (mm/season)
        'rain_low_c', 'rain_med_c', 'rain_high_c', 'rain_width',
        # humidity (%)
        'hum_low_c', 'hum_med_c', 'hum_high_c', 'hum_width',
        # wind speed (km/h)
        'wind_low_c', 'wind_med_c', 'wind_high_c', 'wind_width',
        # soil temperature (°C)
        'soilt_low_c', 'soilt_med_c', 'soilt_high_c',
    ]

    # Default values calibrated from vellore_merged_final.csv stats
    _DEFAULTS = {
        'area_low_c':   500,   'area_med_c':   5000,  'area_high_c':  50000, 'area_width':  3000,
        'yield_low_c':  10,    'yield_med_c':  40,    'yield_high_c': 75,    'yield_width': 20,
        'temp_low_c':   22,    'temp_med_c':   28,    'temp_high_c':  35,    'temp_width':  5,
        'rain_low_c':   200,   'rain_med_c':   600,   'rain_high_c':  1100,  'rain_width':  200,
        'hum_low_c':    55,    'hum_med_c':    66,    'hum_high_c':   76,    'hum_width':   6,
        'wind_low_c':   7.5,   'wind_med_c':   9.5,   'wind_high_c':  12.0,  'wind_width':  1.5,
        'soilt_low_c':  26,    'soilt_med_c':  29,    'soilt_high_c': 33,
    }

    # Bounds for each parameter (used by optimisers)
    _BOUNDS = [
        (100,    10000),   # area_low_c
        (1000,   30000),   # area_med_c
        (10000, 100000),   # area_high_c
        (500,    10000),   # area_width
        (1,      30),      # yield_low_c
        (20,     60),      # yield_med_c
        (50,     95),      # yield_high_c
        (5,      40),      # yield_width
        (15,     26),      # temp_low_c
        (24,     31),      # temp_med_c
        (30,     42),      # temp_high_c
        (2,      10),      # temp_width
        (50,    400),      # rain_low_c
        (300,   800),      # rain_med_c
        (700,  1600),      # rain_high_c
        (50,   400),       # rain_width
        (50,    65),       # hum_low_c
        (60,    72),       # hum_med_c
        (70,    85),       # hum_high_c
        (2,     10),       # hum_width
        (7.0,   9.5),      # wind_low_c
        (8.5,  11.5),      # wind_med_c
        (10.5,  14.5),     # wind_high_c
        (0.5,   3.0),      # wind_width
        (25,    28),       # soilt_low_c
        (27,    31),       # soilt_med_c
        (30,    35),       # soilt_high_c
    ]

    def __init__(self, params_vector=None):
        self.defaults = dict(self._DEFAULTS)
        if params_vector is not None:
            self._apply_vector(params_vector)

    def _apply_vector(self, v):
        for i, k in enumerate(self._KEYS[:len(v)]):
            self.defaults[k] = float(v[i])

    def as_vector(self):
        return np.array([self.defaults[k] for k in self._KEYS], dtype=float)

    @staticmethod
    def bounds():
        return list(FuzzyParams._BOUNDS)

    @property
    def dim(self):
        return len(self._KEYS)


# ─────────────────────────────────────────────────────────────────
# FUZZY INFERENCE SYSTEM
# ─────────────────────────────────────────────────────────────────

class QuantumAgroFIS:
    """
    Extended Mamdani FIS with 8 inputs.
    Inputs : area, yield_history, temperature, rainfall,
             humidity, wind_speed, soil_temp, soil_moisture (scalar gate)
    Output : suitability score 0-100 per crop → top-5 recommendation
    """

    def __init__(self, params: FuzzyParams = None):
        self.p = params if params else FuzzyParams()

    # ── Fuzzification helpers ─────────────────────────────────────

    def _fuzz3(self, x, low_c, med_c, high_c, width, x_min=0, x_max=1e9):
        """Generic 3-label triangular fuzzifier."""
        d = self.p.defaults
        low  = trimf(x, [x_min,  low_c,  low_c  + width])
        med  = trimf(x, [low_c,  med_c,  med_c  + width])
        high = trimf(x, [med_c,  high_c, x_max])
        return {'low': low, 'med': med, 'high': high}

    def fuzzify_area(self, area):
        d = self.p.defaults
        return self._fuzz3(area,
                           d['area_low_c'], d['area_med_c'], d['area_high_c'],
                           d['area_width'], x_min=0, x_max=200000)

    def fuzzify_yield(self, yield_norm):
        d = self.p.defaults
        return self._fuzz3(yield_norm,
                           d['yield_low_c'], d['yield_med_c'], d['yield_high_c'],
                           d['yield_width'], x_min=0, x_max=100)

    def fuzzify_temp(self, temp):
        d = self.p.defaults
        return self._fuzz3(temp,
                           d['temp_low_c'], d['temp_med_c'], d['temp_high_c'],
                           d['temp_width'], x_min=10, x_max=50)

    def fuzzify_rain(self, rain_mm):
        d = self.p.defaults
        return self._fuzz3(rain_mm,
                           d['rain_low_c'], d['rain_med_c'], d['rain_high_c'],
                           d['rain_width'], x_min=0, x_max=2000)

    def fuzzify_humidity(self, hum):
        d = self.p.defaults
        return self._fuzz3(hum,
                           d['hum_low_c'], d['hum_med_c'], d['hum_high_c'],
                           d['hum_width'], x_min=40, x_max=100)

    def fuzzify_wind(self, wind):
        d = self.p.defaults
        return self._fuzz3(wind,
                           d['wind_low_c'], d['wind_med_c'], d['wind_high_c'],
                           d['wind_width'], x_min=5, x_max=20)

    def fuzzify_soilt(self, soilt):
        d = self.p.defaults
        # No width param for soil_temp — use fixed 2°C
        low  = trimf(soilt, [20, d['soilt_low_c'], d['soilt_low_c'] + 3])
        med  = trimf(soilt, [d['soilt_low_c'], d['soilt_med_c'], d['soilt_med_c'] + 3])
        high = trimf(soilt, [d['soilt_med_c'], d['soilt_high_c'], 40])
        return {'low': low, 'med': med, 'high': high}

    # ── Rule base ────────────────────────────────────────────────

    def _crop_suitability(self, crop_idx, area_mf, yield_mf, temp_mf,
                          rain_mf, hum_mf, wind_mf, soilt_mf,
                          soil_moist_raw, season):
        """
        Extended rule base — 10 rules incorporating all 8 inputs.
        Returns defuzzified suitability score [0, 100].
        """
        affinity     = SEASON_AFFINITY.get(season, [1]*10)[crop_idx]
        season_score = affinity / 2.0   # 0.0, 0.5, 1.0

        # ── Original rules (area × yield × temp) ─────────────────
        r1 = min(area_mf['high'], yield_mf['high']) * 0.9          # high area + high yield
        r2 = min(area_mf['med'],  yield_mf['med'])  * 0.6          # medium suitability
        r3 = min(area_mf['low'],  yield_mf['low'])  * 0.2          # low suitability
        r4 = temp_mf['med'] * 0.7                                   # moderate temp boost
        r5 = (temp_mf['high'] * 0.6
              if crop_idx in HEAT_LOVERS
              else (1 - temp_mf['high']) * 0.5)                     # temp affinity
        r6 = min(area_mf['high'], yield_mf['high'], temp_mf['med']) * 1.0  # triple high

        # ── New rules (rainfall × humidity × wind × soil) ─────────
        # Rule 7: rain-loving crops rewarded for high rainfall + humidity
        if crop_idx in RAIN_LOVERS:
            r7 = min(rain_mf['high'], hum_mf['high']) * 0.85
        else:
            # Drought-hardy crops prefer low-medium rain
            r7 = min(rain_mf['low'], rain_mf['med']) * 0.5 + \
                 (1 - rain_mf['high']) * 0.35

        # Rule 8: moderate humidity is generally best for all crops
        r8 = hum_mf['med'] * 0.6

        # Rule 9: calm wind is preferred (high wind hurts most crops)
        r9 = (1 - wind_mf['high']) * 0.5

        # Rule 10: soil temperature moderate boost
        r10 = soilt_mf['med'] * 0.65

        # Rule 11: soil moisture gate (raw 0-1 value, 0.2 typical for Vellore)
        # Crops needing more moisture get a boost when soil_moist > 0.22
        if crop_idx in RAIN_LOVERS:
            soil_gate = float(np.clip(soil_moist_raw / 0.25, 0, 1))
        else:
            soil_gate = float(np.clip(0.3 / (soil_moist_raw + 0.05), 0, 1))

        # ── Aggregation ──────────────────────────────────────────
        # Weighted sum; denominator chosen so max ≈ 1 with all rules firing
        aggregate = (r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10) / 6.5

        # Season gate × soil gate
        final = aggregate * (0.25 + 0.55 * season_score) * (0.8 + 0.2 * soil_gate)

        return float(np.clip(final * 100, 0, 100))

    # ── Main inference ────────────────────────────────────────────

    def predict(self, area, yield_history, temperature,
                rainfall, humidity, wind_speed, soil_temp, soil_moisture,
                season):
        """
        Returns top-5 crop recommendations with scores, yield & production estimates.

        Parameters
        ----------
        area          : cultivated area (ha)
        yield_history : recent average yield (tons/ha or crop-specific unit)
        temperature   : season average temperature (°C)
        rainfall      : season total rainfall (mm)
        humidity      : season average relative humidity (%)
        wind_speed    : season average wind speed (km/h)
        soil_temp     : season average soil temperature (°C)
        soil_moisture : season average volumetric soil moisture (0-1)
        season        : one of Kharif / Rabi / Whole Year / Summer / Autumn / Winter
        """
        area_mf  = self.fuzzify_area(area)
        temp_mf  = self.fuzzify_temp(temperature)
        rain_mf  = self.fuzzify_rain(rainfall)
        hum_mf   = self.fuzzify_humidity(humidity)
        wind_mf  = self.fuzzify_wind(wind_speed)
        soilt_mf = self.fuzzify_soilt(soil_temp)

        results = []
        for i, crop in enumerate(TOP_CROPS):
            yn       = min((yield_history / YIELD_NORM_MAX[crop]) * 100, 100)
            yield_mf = self.fuzzify_yield(yn)

            score    = self._crop_suitability(
                i, area_mf, yield_mf, temp_mf,
                rain_mf, hum_mf, wind_mf, soilt_mf,
                soil_moisture, season
            )
            yield_est = CROP_YIELD_BASE[crop] * (0.5 + 0.5 * score / 100)
            prod_est  = area * yield_est

            results.append({
                'crop':      crop,
                'score':     round(score, 2),
                'yield_est': round(yield_est, 2),
                'prod_est':  round(prod_est, 2),
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:5]


# ─────────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────

def evaluate_fis(fis, vellore_df):
    """
    Top-5 hit rate on Vellore eval set.
    Returns (accuracy, mean_top1_score, n_evaluated).
    Uses all 8 input features from vellore_merged_final.csv.
    """
    hits  = 0
    total = 0
    scores_top1 = []

    for _, row in vellore_df.iterrows():
        if row['crop'] not in TOP_CROPS:
            continue
        preds = fis.predict(
            area          = row['area'],
            yield_history = row['yield'],
            temperature   = row['season_temp_avg'],
            rainfall      = row['season_rainfall_mm'],
            humidity      = row['season_humidity_avg'],
            wind_speed    = row['season_wind_speed'],
            soil_temp     = row['season_soil_temp'],
            soil_moisture = row['season_soil_moist'],
            season        = row['season'],
        )
        top5_names = [p['crop'] for p in preds]
        if row['crop'] in top5_names:
            hits += 1
        scores_top1.append(preds[0]['score'])
        total += 1

    accuracy = hits / total if total > 0 else 0.0
    return accuracy, float(np.mean(scores_top1)) if scores_top1 else 0.0, total
