# 🌾 QuantumAgro — Crop Recommendation System

A quantum-inspired fuzzy inference system for intelligent crop recommendation using environmental, climatic, and soil parameters.
This project combines fuzzy logic + metaheuristic optimization + machine learning benchmarking to deliver accurate, interpretable agricultural predictions.

---

## 🚀 Overview

QuantumAgro is an advanced decision-support system that:

- Recommends Top-5 suitable crops
- Uses 8 real-world agricultural inputs
- Applies fuzzy logic for interpretability
- Enhances performance using:
  - Genetic Algorithm (GA)
  - Particle Swarm Optimization (PSO)
  - Quantum PSO (QPSO)

---

## 🧠 Key Features

- Fuzzy Inference System (FIS) with 27 parameters
- Optimization using GA, PSO, and QPSO
- Dual ML Benchmarking (fair vs unfair comparison)
- Bayesian hyperparameter tuning
- Interactive Streamlit dashboard

---

## 📥 Input Parameters

- Area (hectares)
- Yield history
- Temperature
- Rainfall
- Humidity
- Wind speed
- Soil temperature
- Soil moisture

---

## 📊 Model Pipeline

Fuzzy System → GA → PSO → QPSO (final model)
        ↓
Bayesian Optimization
        ↓
ML Benchmark Comparison

---

## 🏆 Performance Highlights

- Fuzzy Baseline: ~85.5% Top-5 Accuracy
- GA Optimized: ~90.8%
- PSO Optimized: ~89.4%
- QPSO Optimized (Best): ~91.5%

---

## 📂 Project Structure

QuantumAgro/
|
|-- app.py                  # Streamlit dashboard
|-- train.py                # Full training pipeline
|-- fuzzy_engine.py         # Fuzzy inference system
|-- genetic_algorithm.py    # GA optimizer
|-- pso.py                  # PSO optimizer
|-- qpso.py                 # Quantum PSO
|-- bayesian_tuner.py       # Hyperparameter tuning
|-- ml_benchmark.py         # ML comparisons
|-- figures.py              # Visualization generator
|
|-- crop_dataset/
|   |-- crops_data.csv
|   |-- vellore_merged_final.csv
|
|-- results/
|   |-- *.pkl
|   |-- *.json
|   |-- figures/
|
|-- README.md

---

## ▶️ How to Run

1. Install dependencies:
pip install numpy pandas matplotlib scikit-learn streamlit scikit-optimize

2. Train the model:
python train.py

3. Run the dashboard:
streamlit run app.py

---

