# BioAI Project 3 - Feature Selection Optimization

This project implements evolutionary algorithms for feature selection in machine learning. It uses lookup tables containing pre-computed classification errors for all possible feature combinations, then applies optimization algorithms to find the best feature subsets.

## Project Structure

```
bioai-project3/
├── src/
│   ├── algorithms/
│   │   ├── base.py           # Abstract base class for optimization algorithms
│   │   ├── genetic_algorithm.py  # Single-objective Genetic Algorithm
│   │   ├── nsga2.py          # Multi-objective NSGA-II
│   │   └── pso.py            # Particle Swarm Optimization
│   ├── table/
│   │   ├── creation.py      # Lookup table generation from datasets
│   │   └── analysis.py       # Visualization tools
│   └── utils/
│       └── helpers.py         # Shared genetic operators and utilities
├── outputs/                   # Generated lookup tables and plots
├── main.py                   # Entry point
└── requirements.txt           # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Creating Lookup Tables

First, create lookup tables from datasets:

```python
import pandas as pd
from src.table.creation import get_table, export_to_csv

# Load your dataset
dataset = pd.read_csv("your_dataset.csv")
feature_columns = dataset.columns[:-1].tolist()  # All except target
y = dataset.target

# Create lookup table
table = get_table(dataset, feature_columns, y, penalty_factor=0.01)

# Save to CSV
export_to_csv(table, "dataset_name")
```

### Running Optimization Algorithms

```python
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.nsga2 import NSGA2
from src.algorithms.pso import ParticleSwarmOptimization
from src.table.creation import csv_to_dict

# Load pre-computed lookup table
table = csv_to_dict("outputs/wine_complete_table.csv")

# Genetic Algorithm
ga = GeneticAlgorithm()
best_individual, best_fitness = ga.run(
    table,
    population_size=100,
    generations=500,
    tournament_size=6,
    mutation_rate=0.01,
    elite_frac=0.2,
    verbose=True
)

# Particle Swarm Optimization
pso = ParticleSwarmOptimization()
best_individual, best_fitness = pso.run(
    table,
    num_particles=50,
    num_iterations=50,
    cognitive_param=1.5,
    social_param=1.5,
    verbose=True
)

# NSGA-II (Multi-objective)
nsga = NSGA2()
ranks, fronts, population = nsga.run(
    table,
    population_size=100,
    generations=100,
    mutation_rate=0.01,
    verbose=True
)
```

### Using the Main Entry Point

```python
from main import run_genetic_algorithm, run_pso, load_table

# Load tables
table = load_table("wine")

# Run algorithms
result = run_genetic_algorithm(table, verbose=True)
```

## Algorithms

### Genetic Algorithm (GA)
- Tournament selection
- Single-point crossover
- Bit-flip mutation
- Elitism selection

### NSGA-II
- Non-dominated sorting
- Crowding distance calculation
- Binary tournament selection based on rank and crowding distance
- Minimizes both error and number of features

### Particle Swarm Optimization (PSO)
- Binary PSO with sigmoid velocity transfer
- Inertia weight decreasing over iterations
- Cognitive and social components

## Visualization

```python
from src.table.analysis import visualization_2d, hinged_bitstring_map

# 2D visualization of the search space
visualization_2d(table, "dataset_name")

# Hinged bitstring map
hinged_bitstring_map(table, "dataset_name")
```

## Requirements

- numpy
- pandas
- scikit-learn
- tqdm
- matplotlib

## Datasets

The project has been tested with:
- Wine (13 features)
- Glass (9 features)
- Magic (10 features)
- Heart Diseases
- Zoo
- Letters
