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

In this repository, we create Lookup table beforehand to get the output of our genetic algorithms faster. In practice, we would run predictions on every new individidual of the population (each individual being a bitstring representing a feature combination).

First, create lookup tables from datasets using table_creation.py. You will require a .csv dataset with columns included, and the target variable 
named as 'Class' (see wine.csv for example). Once you have a lookup table created in /outputs folder, you can run main.py to find the best feature combination on your table(s).

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

## Visualization (to be refactored)

```python
from src.table.analysis import visualization_2d, hinged_bitstring_map

# 2D visualization of the search space
visualization_2d(table, "dataset_name")

# Hinged bitstring map
hinged_bitstring_map(table, "dataset_name")
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
- 
## Results

### Datasets & Setup
Three UCI datasets were used (Glass: 9 features, Wine: 13 features, 
Magic: 10 features), with a Decision Tree Classifier trained on a 70/30 
split. Lookup tables were precomputed over all 2^n feature subsets, 
serving as a cheap fitness function for the three algorithms.

### Landscape Analysis
| Dataset | Local Optima | Global Minimum Value | Best Feature Subset |
|---------|-------------|---------------------|-------------------|
| Glass   | 7           | 0.2862              | 110100100         |
| Wine    | 163         | 0.0670              | 1000101000000     |
| Magic   | 3           | 0.2051              | 1000000010        |

Wine's landscape (163 local optima) proved significantly harder to 
navigate than Magic's (3 local optima), which was reflected in algorithm 
performance.

### Algorithm Comparison (10 independent runs each)
| Algorithm | Glass | Wine | Magic | Notes |
|-----------|-------|------|-------|-------|
| PSO  | 0.286 ± 0.000 | 0.067 ± 0.000 | 0.205 ± 0.000 | 100% global optimum rate on all 3 |
| SGA  | 0.298 ± 0.019 | 0.071 ± 0.006 | 0.205 ± 0.000 | Higher variance on multimodal landscapes |
| NSGA-II | 0.286 ± 0.000 | 0.067 ± 0.000 | 0.205 ± 0.000 | 100% global optimum rate on all 3 |

**Key finding:** PSO and NSGA-II reliably found the global optimum across 
all three landscapes. SGA struggled on Wine (the most multimodal landscape 
with 163 local optima), achieving only 70% success rate and higher fitness 
variance — consistent with the No Free Lunch theorem: SGA's greedy 
selection pressure is a liability on rugged landscapes.

### Blind Test Instances (Task 6)
A Random Forest Classifier was used on three unseen datasets 
(Heart, Zoo, Letters).

| Dataset | Best Individual Found | Best Fitness | Notes |
|---------|----------------------|--------------|-------|
| Heart   | 010100110111         | 0.3111       | Hamming distance 0 from personal optimum |
| Zoo     | 000000001011100      | 0.0781       | Hamming distance 0 from personal optimum |
| Letters | 000000110001010      | 0.7768       | Exact match with reference solution |

All three global optima were reached by the algorithms. Results diverged 
from the reference solutions on Heart and Zoo, likely due to differences 
in Random Forest initialization or train/test split randomness.
