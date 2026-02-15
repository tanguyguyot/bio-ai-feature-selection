import random
import time
from typing import Any

from tqdm import tqdm

from src.algorithms.base import BaseAlgorithm
from src.utils import helpers


class GeneticAlgorithm(BaseAlgorithm):
    def run(
        self,
        lookup_dict: dict,
        population_size: int = 100,
        generations: int = 500,
        tournament_size: int = 6,
        mutation_rate: float = 0.01,
        elite_frac: float = 0.2,
        verbose: bool = False,
        **kwargs
    ) -> tuple[str, float]:
        features_amount = self.get_feature_count(lookup_dict)
        population = [
            helpers.generate_individual(features_amount)
            for _ in range(population_size)
        ]

        start_time = time.time()
        best_fitness = min(
            helpers.get_fitness(ind, lookup_dict) for ind in population
        )
        if verbose:
            print(f"Initial best fitness: {best_fitness}")

        for generation in tqdm(range(generations), desc="GA Generation"):
            children = []
            for _ in range(population_size // 2):
                parent1, parent2 = helpers.tournament_selection(
                    population, lookup_dict, tournament_size
                )
                child1, child2 = helpers.crossover(parent1, parent2)
                children.extend(
                    [
                        helpers.mutate(child1, mutation_rate),
                        helpers.mutate(child2, mutation_rate),
                    ]
                )
            population = self._elite_selection(
                children, lookup_dict, amount=round(population_size * elite_frac)
            )
            best_fitness = min(
                helpers.get_fitness(ind, lookup_dict) for ind in population
            )

            if generation % 100 == 0 and verbose:
                tqdm.write(
                    f"Generation {generation} ; best = {best_fitness}"
                )

        best_individual = min(
            population, key=lambda x: helpers.get_fitness(x, lookup_dict)
        )
        best_fitness = helpers.get_fitness(best_individual, lookup_dict)

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Best individual: {best_individual}, Fitness: {best_fitness}")
            print(
                f"Time taken: {elapsed_time:.2f} seconds for {generations} generations"
            )

        return best_individual, best_fitness

    def _elite_selection(
        self, population: list, lookup_dict: dict, amount: int
    ) -> list:
        population.sort(key=lambda x: helpers.get_fitness(x, lookup_dict))
        elite_population = population[:amount]
        remaining_population = population[amount:]
        random.shuffle(remaining_population)
        remaining_amount = len(population) - amount
        return elite_population + remaining_population[:remaining_amount]
