from abc import ABC, abstractmethod
from typing import Any


class BaseAlgorithm(ABC):
    @abstractmethod
    def run(self, lookup_dict: dict, **kwargs) -> Any:
        pass

    @staticmethod
    def get_feature_count(lookup_dict: dict) -> int:
        return len(next(iter(lookup_dict.keys())))
