# cuckoo_search_experiments.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math

from collections import defaultdict

# Define benchmark functions
def sphere(x): return np.sum(x**2)
def rastrigin(x): return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
def ackley(x):
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e
def rosenbrock(x): return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# Lévy flight step
def levy_flight(Lambda):
    sigma = (math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.randn()
    v = np.random.randn()
    step = u * sigma / abs(v) ** (1 / Lambda)
    return step


# Cuckoo Search Algorithm
def cuckoo_search(obj_func, n=20, d=10, pa=0.25, n_iter=500):
    nests = np.random.uniform(-5, 5, (n, d))
    fitness = np.apply_along_axis(obj_func, 1, nests)
    best = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    history = [best_fitness]

    for _ in range(n_iter):
        new_nests = np.copy(nests)
        for i in range(n):
            step_size = levy_flight(1.5)
            new_nests[i] += step_size * (nests[i] - best)
            new_nests[i] = np.clip(new_nests[i], -5, 5)
        new_fitness = np.apply_along_axis(obj_func, 1, new_nests)
        for i in range(n):
            if new_fitness[i] < fitness[i]:
                nests[i] = new_nests[i]
                fitness[i] = new_fitness[i]
        K = np.random.rand(n, d) < pa
        nests[K] = np.random.uniform(-5, 5, (np.sum(K), ))
        current_best = nests[np.argmin(fitness)]
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best = current_best
            best_fitness = current_best_fitness
        history.append(best_fitness)

    return best, best_fitness, history

# Run experiments
functions = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Ackley": ackley,
    "Rosenbrock": rosenbrock
}

results = []
for dim in [10, 30]:
    for name, func in functions.items():
        print(f"Running {name} Function in {dim}D...")
        best_sol, best_val, history = cuckoo_search(func, d=dim, n_iter=500)
        result = {
            "Function": name,
            "Dimension": dim,
            "Best Fitness": best_val,
            "Best Solution": best_sol.tolist(),
            "History": history
        }
        results.append(result)

# Save CSV
rows = []
for r in results:
    row = {
        "Function": r["Function"],
        "Dimension": r["Dimension"],
        "Best Fitness": r["Best Fitness"],
        "Best Solution": r["Best Solution"]
    }
    history = {f"history_{i}": val for i, val in enumerate(r["History"])}
    row.update(history)
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("cuckoo_search_results.csv", index=False)
print("✅ Results saved to 'cuckoo_search_results.csv'")

# Plot comparison figures
os.makedirs("plots", exist_ok=True)
func_results = defaultdict(dict)
for r in results:
    func_results[r["Function"]][r["Dimension"]] = r["History"]

for func_name, histories in func_results.items():
    plt.figure(figsize=(8, 5))
    for dim, hist in histories.items():
        plt.plot(hist, label=f"{dim}D")
    plt.title(f"Cuckoo Search Convergence: {func_name}")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    filename = f"plots/{func_name}_comparison.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

print("✅ Comparison plots saved to 'plots/' directory.")
