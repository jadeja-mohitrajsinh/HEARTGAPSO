import numpy as np


def repair_rejected(rejected, evaluate, dimensions: int, seed: int, iterations: int = 20,
                    inertia: float = 0.65, cognitive: float = 1.4, social: float = 1.4):
    """Improve every rejected GA individual and return each particle's personal best."""
    if not rejected:
        return [], []
    rng = np.random.default_rng(seed)
    positions = np.array([item[0] for item in rejected], dtype=float)
    velocities = rng.normal(0, 0.08, positions.shape)
    scores = np.array([item[1] for item in rejected], dtype=float)
    personal = positions.copy()
    personal_scores = scores.copy()
    personal_metrics = [None] * len(rejected)
    global_index = int(np.argmax(scores))
    global_best = positions[global_index].copy()
    global_score = float(scores[global_index])

    stagnant_iterations = 0
    for iteration in range(iterations):
        if iteration > 0:
            print(
                f"[PSO] Iteration {iteration + 1}/{iterations} | "
                f"particles={len(rejected)} | global_best={global_score:.4f}",
                flush=True,
            )
        random_a = rng.random(positions.shape)
        random_b = rng.random(positions.shape)
        velocities = (
            inertia * velocities
            + cognitive * random_a * (personal - positions)
            + social * random_b * (global_best - positions)
        )
        positions = np.clip(positions + velocities, 0.0, 1.0)
        iteration_best = global_score
        evaluated_positions = set()
        for index, position in enumerate(positions):
            binary_position = position.copy()
            probability = 1.0 / (1.0 + np.exp(-position[:dimensions]))
            binary_position[:dimensions] = (rng.random(dimensions) < probability).astype(float)
            if not binary_position[:dimensions].any():
                binary_position[rng.integers(0, dimensions)] = 1.0
            key = tuple(np.round(binary_position, 6))
            if key in evaluated_positions:
                continue
            evaluated_positions.add(key)
            score, metrics = evaluate(binary_position[:dimensions], binary_position[dimensions:])
            scores[index] = score
            if score > personal_scores[index]:
                personal[index] = binary_position.copy()
                personal_scores[index] = score
                personal_metrics[index] = metrics
            if score > global_score:
                global_best, global_score = binary_position.copy(), score
        if global_score <= iteration_best + 1e-10:
            stagnant_iterations += 1
        else:
            stagnant_iterations = 0
        if stagnant_iterations >= 3:
            print(
                f"[PSO] Early stopping after {iteration + 1} stagnant iterations",
                flush=True,
            )
            break

    results = []
    history = []
    for index in range(len(personal)):
        metrics = personal_metrics[index]
        if metrics is None:
            _, metrics = evaluate(personal[index][:dimensions], personal[index][dimensions:])
        results.append((personal[index], float(personal_scores[index]), metrics))
        history.append({
            "particle_id": index,
            "source_ga_candidate_id": index,
            "initial_position": rejected[index][0].tolist(),
            "initial_fitness": float(rejected[index][1]),
            "velocity": velocities[index].tolist(),
            "updated_position": personal[index].tolist(),
            "updated_fitness": float(personal_scores[index]),
            "improvement": float(personal_scores[index] - rejected[index][1]),
        })
    return results, history
