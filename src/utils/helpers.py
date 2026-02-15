import random
import numpy as np


def generate_individual(length: int) -> str:
    return "".join(random.choice("01") for _ in range(length))


def generate_individual_numpy(length: int) -> np.ndarray:
    return np.random.randint(0, 2, length)


def mutate(bitstring: str, mutation_rate: float) -> str:
    out = []
    for bit in bitstring:
        if random.random() < mutation_rate:
            out.append("1" if bit == "0" else "0")
        else:
            out.append(bit)
    return "".join(out)


def mutate_numpy(bitstring: np.ndarray, mutation_rate: float) -> np.ndarray:
    rand = np.random.rand(bitstring.size)
    mask = rand < mutation_rate
    result = bitstring.copy()
    result[mask] = 1 - result[mask]
    return result


def crossover(parent1: str, parent2: str) -> tuple[str, str]:
    index = random.randint(1, len(parent1) - 1)
    child1 = parent1[:index] + parent2[index:]
    child2 = parent2[:index] + parent1[index:]
    return child1, child2


def crossover_uniform(parent1: str, parent2: str) -> tuple[str, str]:
    child1 = "".join(random.choice([a, b]) for a, b in zip(parent1, parent2))
    child2 = "".join(random.choice([a, b]) for a, b in zip(parent1, parent2))
    return child1, child2


def tournament_selection(
    population: list, lookup_dict: dict, tournament_size: int
) -> tuple[str, str]:
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=lambda x: get_fitness(x, lookup_dict))
    return tournament[0], tournament[1]


def get_fitness(bitstring: str, lookup_dict: dict) -> float:
    return lookup_dict.get(bitstring, {"Lookup value": np.inf}).get("Lookup value", np.inf)


def get_error(bitstring: str, lookup_dict: dict) -> float:
    return lookup_dict.get(bitstring, {"Error": np.inf}).get("Error", np.inf)


def count_features(bitstring: str) -> int:
    return bitstring.count("1")


def hamming_distance(x: str, y: str) -> int:
    assert len(x) == len(y), "Strings must be of the same length"
    return sum(el1 != el2 for el1, el2 in zip(x, y))


def bitstring_to_numpy(bitstring: str) -> np.ndarray:
    return np.array([int(b) for b in bitstring])


def numpy_to_bitstring(arr: np.ndarray) -> str:
    return "".join(str(int(gene)) for gene in arr)
