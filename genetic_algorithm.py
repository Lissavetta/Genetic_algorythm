import numpy as np
import random
import matplotlib.pyplot as plt

class GeneticAlgorithmTSP:
    def __init__(self, distance_matrix, population_size=100, mutation_rate=0.01, generations=500):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations

    def create_initial_population(self):
        population = []
        for _ in range(self.population_size):
            individual = list(range(self.num_cities))
            random.shuffle(individual)
            population.append(individual)
        return population

    def fitness(self, individual):
        total_distance = 0
        for i in range(len(individual)):
            total_distance += self.distance_matrix[individual[i]][individual[(i + 1) % len(individual)]]
        return total_distance

    def selection(self, population):
        sorted_population = sorted(population, key=self.fitness)
        return sorted_population[:self.population_size // 2]

    def crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(self.num_cities), 2))
        child = [None] * self.num_cities
        child[start:end] = parent1[start:end]

        current_position = end
        for gene in parent2:
            if gene not in child:
                child[current_position] = gene
                current_position = (current_position + 1) % self.num_cities
        
        return child

    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(individual) - 1)
                individual[i], individual[j] = individual[j], individual[i]

    def run(self):
        population = self.create_initial_population()
        
        for generation in range(self.generations):
            selected_parents = self.selection(population)
            next_generation = selected_parents.copy()

            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(selected_parents, 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                next_generation.append(child)

            population = next_generation

        best_individual = min(population, key=self.fitness)
        best_distance = self.fitness(best_individual)

        return best_individual, best_distance
    
    
def plot_route(route, distance_matrix):
    # Получаем координаты городов из матрицы расстояний
    cities = np.array([(i, j) for i in range(len(distance_matrix)) for j in range(len(distance_matrix)) if i != j])
    
    # Для упрощения примера используем случайные координаты для городов
    np.random.seed(0)  # Для воспроизводимости
    coords = np.random.rand(len(distance_matrix), 2) * 100

    # Создание графика
    plt.figure(figsize=(10, 6))
    
    # Рисуем маршрут
    route_coords = coords[route + [route[0]]]  # Добавляем начальный город в конец маршрута
    plt.plot(route_coords[:, 0], route_coords[:, 1], marker='o')
    
    # Подписываем города
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=12)

    plt.title('Оптимальный маршрут (Генетический алгоритм)')
    plt.xlabel('Координаты X')
    plt.ylabel('Координаты Y')
    plt.grid()
    plt.show()

def main():
    # Пример матрицы расстояний (симметричная)
    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ])

    ga_tsp = GeneticAlgorithmTSP(distance_matrix)
    best_route, best_distance = ga_tsp.run()

    print("Лучший маршрут:", best_route)
    print("Лучшее расстояние:", best_distance)

    # Построение графика маршрута
    plot_route(best_route, distance_matrix)

if __name__ == "__main__":
    main()
