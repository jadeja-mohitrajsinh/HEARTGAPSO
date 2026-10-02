"""Fitness configuration for the GAPSO Random Forest search."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FitnessConfig:
    folds: int = 5
    repeats: int = 2
    roc_auc_weight: float = 0.40
    f1_weight: float = 0.25
    sensitivity_weight: float = 0.20
    specificity_weight: float = 0.15
    feature_penalty: float = 0.002
    stability_penalty: float = 0.02

    def __post_init__(self) -> None:
        if self.folds < 3:
            raise ValueError("At least three stratified folds are required")
        if self.repeats < 1:
            raise ValueError("CV repeats must be positive")
        if min(self.roc_auc_weight, self.f1_weight, self.sensitivity_weight, self.specificity_weight) < 0:
            raise ValueError("Fitness weights must be non-negative")
