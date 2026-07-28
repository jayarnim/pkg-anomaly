from abc import ABC, abstractmethod


class Annealing(ABC):
    @abstractmethod
    def __call__(
        self, 
        step: int,
    ) -> float:
        raise NotImplementedError