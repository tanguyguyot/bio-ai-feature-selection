"""
Main entry point for the feature selection optimization algorithms.

This module provides a unified interface to run different optimization algorithms
(Genetic Algorithm, NSGA-II, Particle Swarm Optimization) on feature selection tasks.
"""

from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.nsga2 import NSGA2
from src.algorithms.pso import ParticleSwarmOptimization
from src.table.creation import csv_to_dict
from src.utils.helpers import hamming_distance


def run_genetic_algorithm(lookup_dict: dict, **kwargs):
    ga = GeneticAlgorithm()
    return ga.run(lookup_dict, **kwargs)


def run_nsga2(lookup_dict: dict, **kwargs):
    nsga = NSGA2()
    return nsga.run(lookup_dict, **kwargs)


def run_pso(lookup_dict: dict, **kwargs):
    pso = ParticleSwarmOptimization()
    return pso.run(lookup_dict, **kwargs)


def load_table(dataset_name: str, output_dir: str = "outputs") -> dict:
    return csv_to_dict(f"{output_dir}/{dataset_name}_complete_table.csv")


if __name__ == "__main__":
    datasets = ["glass", "wine", "magic", "heart_diseases", "zoo", "letters"]

    print("Loading lookup tables...")
    tables = {name: load_table(name) for name in datasets}

    print("\n=== Running Genetic Algorithm ===")
    for name, table in tables.items():
        best_individual, best_fitness = run_genetic_algorithm(table, verbose=True)
        print(f"{name}: {best_individual} -> {best_fitness}")

    print("\n=== Running PSO ===")
    for name, table in tables.items():
        best_individual, best_fitness = run_pso(table, verbose=True)
        print(f"{name}: {best_individual} -> {best_fitness}")
