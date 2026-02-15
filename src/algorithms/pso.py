from typing import Any

import numpy as np
from tqdm import tqdm

from src.algorithms.base import BaseAlgorithm
from src.utils import helpers


class Particle:
    def __init__(self, feature_length: int, lookup_table: dict):
        self.position: np.ndarray = helpers.generate_individual_numpy(feature_length)
        self.velocity: np.ndarray = np.random.randn(feature_length)
        self.fitness: float = self._get_fitness(self.position, lookup_table)
        self.best_position: np.ndarray = self.position.copy()
        self.best_fitness: float = self.fitness

    def _get_fitness(self, individual: np.ndarray, lookup_table: dict) -> float:
        bitstring = helpers.numpy_to_bitstring(individual)
        return helpers.get_fitness(bitstring, lookup_table)


class ParticleSwarmOptimization(BaseAlgorithm):
    def run(
        self,
        lookup_dict: dict,
        num_particles: int = 50,
        num_iterations: int = 50,
        cognitive_param: float = 1.5,
        social_param: float = 1.5,
        verbose: bool = False,
        **kwargs
    ) -> tuple[str, float]:
        feature_length = self.get_feature_count(lookup_dict)
        particles = [Particle(feature_length, lookup_dict) for _ in range(num_particles)]

        global_best_position, global_best_fitness = self._find_best_individual(
            particles
        )

        for iteration in tqdm(range(num_iterations), desc="PSO"):
            inertia = 1 - (0.6 * iteration / num_iterations)

            for particle in particles:
                particle.fitness = particle._get_fitness(
                    particle.position, lookup_dict
                )
                if particle.fitness < particle.best_fitness:
                    particle.best_position = particle.position.copy()
                    particle.best_fitness = particle.fitness
                if particle.fitness < global_best_fitness:
                    global_best_position = particle.position.copy()
                    global_best_fitness = particle.fitness

            for particle in particles:
                particle.velocity = self._get_velocity(
                    particle, inertia, cognitive_param, social_param, global_best_position
                )
                self._update_particle(particle)

            if iteration % 10 == 0 and verbose:
                print(f"Gen {iteration}, best fitness = {global_best_fitness}")

        if verbose:
            print("Final best fitness:", global_best_fitness)
            print(
                "Final best position:",
                helpers.numpy_to_bitstring(global_best_position),
            )

        return helpers.numpy_to_bitstring(global_best_position), global_best_fitness

    def _find_best_individual(self, particles: list) -> tuple[np.ndarray, float]:
        best_particle = min(particles, key=lambda p: p.fitness)
        return best_particle.position.copy(), best_particle.fitness

    def _get_velocity(
        self,
        particle: Particle,
        inertia: float,
        cognitive_param: float,
        social_param: float,
        global_best: np.ndarray,
        vmax: float = 4.0,
    ) -> np.ndarray:
        r1 = np.random.rand(particle.position.size)
        r2 = np.random.rand(particle.position.size)
        cognitive = cognitive_param * r1 * (particle.best_position - particle.position)
        social = social_param * r2 * (global_best - particle.position)
        velocity = inertia * particle.velocity + cognitive + social
        return np.clip(velocity, -vmax, vmax)

    def _update_particle(self, particle: Particle) -> None:
        probs = 1 / (1 + np.exp(-particle.velocity))
        rand = np.random.rand(particle.position.size)
        particle.position = (rand < probs).astype(int)
