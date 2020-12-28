import numpy as np, \
    matplotlib.pyplot as pyplot

class Population():
    def __init__(self, bag, matr):
        self.bag = bag
        self.parents = []
        self.score = 0
        self.best = None
        self.matr = matr

def init_population(cities, matr, n_population):
    return Population(
        np.asarray([np.random.permutation(cities) for _ in range(n_population)]),
        matr
    )

def fitness(self, gen):
    return sum(
        [
            self.matr[gen[i], gen[i + 1]]
            for i in range(len(gen) - 1)
        ]
    )

def evaluate(self):
    distances = np.asarray(
        [self.fitness(gen) for gen in self.bag]
    )
    self.score = np.min(distances)
    self.best = self.bag[distances.tolist().index(self.score)]
    self.parents.append(self.best)
    if False in (distances[0] == distances):
        distances = np.max(distances) - distances
    return distances / np.sum(distances)

def select(self, k=4):
    fit = self.evaluate()
    while len(self.parents) < k:
        idx = np.random.randint(0, len(fit))
        if fit[idx] > np.random.rand():
            self.parents.append(self.bag[idx])
    self.parents = np.asarray(self.parents)

def swap(gen):
    a, b = np.random.choice(len(gen), 2)
    gen[a], gen[b] = (
        gen[b],
        gen[a],
    )
    return gen

def crossover(self, p_cross=0.1):
    children = []
    count, size = self.parents.shape
    for _ in range(len(self.bag)):
        if np.random.rand() > p_cross:
            children.append(
                list(self.parents[np.random.randint(count, size=1)[0]])
            )
        else:
            parent1, parent2 = self.parents[
                               np.random.randint(count, size=2), :
                               ]
            idx = np.random.choice(range(size), size=2, replace=False)
            start, end = min(idx), max(idx)
            child = [None] * size
            for i in range(start, end + 1, 1):
                child[i] = parent1[i]
            pointer = 0
            for i in range(size):
                if child[i] is None:
                    while parent2[pointer] in child:
                        pointer += 1
                    child[i] = parent2[pointer]
            children.append(child)
    return children

# Мутация изменяет количество генов, как определено аргументом num_mutations.
# Изменения случайны.
def mutate(self, p_cross=0.4, p_mut=0.2):
    next_bag = []
    children = self.crossover(p_cross)
    for child in children:
        if np.random.rand() < p_mut:
            next_bag.append(swap(child))
        else:
            next_bag.append(child)
    return next_bag

# Сам алгоритм
def genetic_algorithm(
        cities,
        matr,
        n_population=800,
        n_iter=200,
        selectivity=0.2,
        p_cross=0.5,
        p_mut=0.2,
        print_interval=200,
        return_history=False,
        verbose=False,
):
    pop = init_population(cities, matr, n_population)
    best = pop.best
    score = float("inf")
    history = []

    for i in range(n_iter):
        pop.select(n_population * selectivity)
        history.append(pop.score)
        if verbose:
            print(f"Поколение {i}: Расстояние: {pop.score}")
        elif i % print_interval == 0:
            print(f"Поколение {i}: Расстояние: {pop.score}")
        if pop.score < score:
            best = pop.best
            score = pop.score
        children = pop.mutate(p_cross, p_mut)
        pop = Population(children, pop.matr)

    if return_history:
        return best, history

    print('\n')
    print("Кратчайший путь:")
    return best

def show_plot(best_outputs):
    pyplot.plot(best_outputs)
    pyplot.xlabel("Итерации")
    pyplot.ylabel("Расстояние")
    pyplot.show()

# Решение задачи нахождения кратчайшего пути через все пункты
# при момощи генетического алгоритма
if __name__ == '__main__':

    cities = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    matr = np.zeros([len(cities), len(cities)])
    best_outputs = []

    for i in range(len(cities)):
        for j in range(len(cities)):
            matr[i][j] = matr[j][i] = np.random.randint(50, 100)
            if (i == j):
                matr[i][j] = 0

    print(matr)
    print('\n')

    pop = init_population(cities, matr, 5)

    Population.fitness = fitness
    Population.evaluate = evaluate

    Population.select = select
    pop.select()

    pop.parents

    Population.crossover = crossover
    Population.mutate = mutate
    pop.mutate()

    best_result = genetic_algorithm(cities,matr,verbose=True)
    print(best_result)