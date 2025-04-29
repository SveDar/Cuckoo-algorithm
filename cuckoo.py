import numpy as np
import matplotlib.pyplot as plt

# Benchmark functions
def sphere(x):
    return np.sum(x**2)

def rastrigin(x):
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley(x):
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.e

def rosenbrock(x):
    return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# Lévy flight function
def levy_flight(Lambda):
    sigma = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) / 
            (np.math.gamma((1 + Lambda) / 2) * Lambda * 2**((Lambda - 1) / 2)))**(1 / Lambda)
    u = np.random.randn() * sigma
    v = np.random.randn()
    step = u / abs(v)**(1 / Lambda)
    return step

# Cuckoo Search core
def cuckoo_search(obj_func, n=15, d=2, pa=0.25, n_iter=100):
    nests = np.random.uniform(-5, 5, (n, d))
    fitness = np.apply_along_axis(obj_func, 1, nests)
    best = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    history = [best_fitness]

    for t in range(n_iter):
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

# Run and compare on all benchmark functions
functions = {
    "Sphere": sphere,
    "Rastrigin": rastrigin,
    "Ackley": ackley,
    "Rosenbrock": rosenbrock
}

plt.figure(figsize=(10, 6))
dim = 10  # Increase dimension for complexity

for name, func in functions.items():
    best_sol, best_val, history = cuckoo_search(func, d=dim, n_iter=200)
    print(f"{name} Function:")
    print(f"  Best solution: {np.round(best_sol, 4)}")
    print(f"  Best fitness: {best_val:.6f}\n")
    plt.plot(history, label=name)

plt.title("Cuckoo Search Convergence on Benchmark Functions")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
