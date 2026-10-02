"""Configuration for the genetic stage of the GAPSO search."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 20
    generations: int = 15
    pso_iterations: int = 3
    max_evaluations: int = 120
    pso_particles: int = 6
    elite_count: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.80
    stagnation_generations: int = 6

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("Population size must be at least four")
        if self.max_evaluations < self.population_size:
            raise ValueError("Evaluation budget must cover one population")
