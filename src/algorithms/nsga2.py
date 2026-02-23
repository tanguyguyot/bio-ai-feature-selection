import random
from typing import Any

import numpy as np
from tqdm import tqdm

from src.algorithms.base import BaseAlgorithm
from src.utils import helpers


class NSGA2(BaseAlgorithm):
    def run(
        self,
        lookup_dict: dict,
        population_size: int = 100,
        generations: int = 100,
        mutation_rate: float = 0.01,
        verbose: bool = False,
        **kwargs
    ) -> tuple[list, list, list]:
        features_amount = self.get_feature_count(lookup_dict)
        population = [
            helpers.generate_individual(features_amount)
            for _ in range(population_size)
        ]
        objectives = [
            (self._objective_error(ind, lookup_dict), helpers.count_features(ind))
            for ind in population
        ]
        ranks, fronts = self._fast_non_dominated_sort(objectives)

        for _ in tqdm(range(generations), desc="NSGA-II"):
            new_population = population.copy()
            for _ in range(len(population) // 2):
                parent1 = self._tournament_selection(
                    population, objectives, ranks, fronts
                )
                parent2 = self._tournament_selection(
                    population, objectives, ranks, fronts
                )
                child1, child2 = helpers.crossover_uniform(parent1, parent2)
                child1 = helpers.mutate(child1, mutation_rate)
                child2 = helpers.mutate(child2, mutation_rate)
                new_population.extend([child1, child2])

            objectives = [
                (self._objective_error(ind, lookup_dict), helpers.count_features(ind))
                for ind in new_population
            ]

            ranks, fronts = self._fast_non_dominated_sort(objectives)
            selected = self._selection(objectives, fronts, population_size)
            population = [new_population[i] for i in selected]

        objectives = [
            (self._objective_error(ind, lookup_dict), helpers.count_features(ind))
            for ind in population
        ]
        ranks, fronts = self._fast_non_dominated_sort(objectives)

        if verbose:
            unique = set(population)
            print(f"Unique individuals: {len(unique)}")
            print(f"Number of fronts: {len(fronts)}")
            print(f"First front size: {len(fronts[0])}")

        return ranks, fronts, population

    def _objective_error(self, bitstring: str, lookup_dict: dict) -> float:
        return helpers.get_error(bitstring, lookup_dict)

    def _non_dominance_condition(
        self, ind1_idx: int, ind2_idx: int, objectives: list
    ) -> bool:
        f1_ind1, f2_ind1 = objectives[ind1_idx]
        f1_ind2, f2_ind2 = objectives[ind2_idx]

        condition1 = all(
            x <= y for x, y in zip((f1_ind1, f2_ind1), (f1_ind2, f2_ind2))
        )
        condition2 = any(
            x < y for x, y in zip((f1_ind1, f2_ind1), (f1_ind2, f2_ind2))
        )
        return condition1 and condition2

    def _fast_non_dominated_sort(self, objectives: list) -> tuple[list, list]:
        len_population = len(objectives)
        fronts = [[]]
        dominated = [[] for _ in range(len_population)]
        dominating = [0 for _ in range(len_population)]
        ranks = [0 for _ in range(len_population)]

        for p in range(len_population):
            for q in range(len_population):
                if self._non_dominance_condition(p, q, objectives):
                    dominated[p].append(q)
                elif self._non_dominance_condition(q, p, objectives):
                    dominating[p] += 1

            if dominating[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated[p]:
                    dominating[q] -= 1
                    if dominating[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return ranks, fronts[:-1]

    def _crowding_distance(self, front: list, objectives: list) -> list:
        distance = [0.0 for _ in front]
        num_objectives = len(objectives[0])

        for i in range(num_objectives):
            sorted_indices = sorted(
                range(len(front)), key=lambda x: objectives[front[x]][i]
            )

            distance[sorted_indices[0]] = float("inf")
            distance[sorted_indices[-1]] = float("inf")

            for j in range(1, len(sorted_indices) - 1):
                max_value = objectives[front[sorted_indices[-1]]][i]
                min_value = objectives[front[sorted_indices[0]]][i]
                distance[sorted_indices[j]] += (
                    objectives[front[sorted_indices[j + 1]]][i]
                    - objectives[front[sorted_indices[j - 1]]][i]
                ) / (max_value - min_value)
        return distance

    def _tournament_selection(
        self, population: list, objectives: list, ranks: list, fronts: list
    ) -> str:
        ind1, ind2 = random.sample(range(len(population)), 2)

        if ranks[ind1] < ranks[ind2]:
            return population[ind1]
        elif ranks[ind1] > ranks[ind2]:
            return population[ind2]

        rank = ranks[ind1]
        front_index1 = fronts[rank].index(ind1)
        front_index2 = fronts[rank].index(ind2)
        distances = self._crowding_distance(fronts[rank], objectives)
        distance1 = distances[front_index1]
        distance2 = distances[front_index2]

        if distance1 > distance2:
            return population[ind1]
        else:
            return population[ind2]

    def _selection(
        self, objectives: list, fronts: list, amount: int
    ) -> list:
        selected = []
        current_rank = 0
        while len(selected) < amount:
            if len(fronts[current_rank]) + len(selected) <= amount:
                selected.extend(fronts[current_rank])
                current_rank += 1
            else:
                distances = self._crowding_distance(fronts[current_rank], objectives)
                sorted_indices = sorted(
                    range(len(fronts[current_rank])),
                    key=lambda x: distances[x],
                    reverse=True,
                )
                selected.extend(
                    [
                        fronts[current_rank][i]
                        for i in sorted_indices[: amount - len(selected)]
                    ]
                )
                break
        return selected
