import numpy as np
import random
from numba import njit, prange
from PIL import Image

size = 256
generations = 1000
resolution = 512
pic = Image.open("orig.jpg")
standard = np.array(pic)


# Crossover of two parents
@njit(parallel=True)
def crossover(a, b):
    return np.asarray((a + b) // 2, dtype=np.uint8)


# Creation of population with random colors
@njit(parallel=True)
def create_population():
    population = np.zeros((size, resolution, resolution, 3), dtype=np.uint8)
    for p in prange(size):
        color1, color2, color3 = [np.random.randint(255), np.random.randint(255), np.random.randint(255)]
        population_part = population[p]
        for i in prange(resolution):
            for j in prange(resolution):
                population_part[i, j, 0], population_part[i, j, 1], population_part[i, j, 2] = color1, color2, color3

    return population


'''
Fitness function for population (smaller result is better result).
Then sorting using fitness function results and creating new population using crossover function.
Doing mutation and returning result
'''
@njit(parallel=True)
def fitness(population):
    deviation = np.zeros(size)
    for q in prange(size):
        for i in prange(512):
            for j in prange(512):
                for k in prange(3):
                    deviation[q] += (population[q][i][j][k] - standard[i][j][k]) ** 2

    sort = np.argsort(deviation)
    new_population = population[sort]
    new_population = new_population[:size // 2]

    for i in range(size // 2, size):
        new_population[i] = crossover(new_population[random.randint(0, size // 2 - 1)], new_population[random.randint(0, size // 2 - 1)])

    makeMutation(new_population)

    return new_population


# Mutation with random size, location and color circles
@njit(parallel=True)
def mutation_1(population_part):
    center = np.array([random.randint(0, 512), random.randint(0, 512)])
    distortion = np.array([random.uniform(1, 4), random.uniform(1, 4)])
    alpha = random.uniform(0.3, 1)
    radius = random.randint(10, 500)
    color = np.array([random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)])

    for i in prange(512):
        for j in prange(512):
            if distortion[0] * (i - center[0]) ** 2 + distortion[1] * (j - center[1]) ** 2 < radius:
                population_part[i, j] = (1 - alpha) * population_part[i, j] + alpha * color

    return population_part


# Mutation with ramdon size, location and color triangles
@njit(parallel=True)
def mutation_2(population_part):
    funct = lambda n, minn, maxn: max(min(maxn, n), minn)

    point0 = (random.randint(0, 512), random.randint(0, 512))
    point1 = (funct(random.randint(point0[0] - 50, point0[0] + 50), 0, 512 - 1), funct(random.randint(point0[1] - 50, point0[1] + 50), 0, 512 - 1))
    point2 = (funct(random.randint(point0[0] - 50, point0[0] + 50), 0, 512 - 1), funct(random.randint(point0[1] - 50, point0[1] + 50), 0, 512 - 1))

    triangle = np.array([point0, point1, point2])
    color = np.array([random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)])
    alpha = random.uniform(0.3, 1)

    for i in range(512):
        for j in range(512):
            if is_mutation2(triangle, (i, j)):
                population_part[i, j] = (1 - alpha) * population_part[i, j] + alpha * color


# Checking is the point in the area of the triangle for the mutation 2
@njit(parallel=True)
def is_mutation2(triangle, p):
    B = (-triangle[1, 1] * triangle[2, 0] + triangle[0, 1] * (-triangle[1, 0] + triangle[2, 0]) + triangle[0, 0] * (
                triangle[1, 1] - triangle[2, 1]) + triangle[1, 0] * triangle[2, 1]) / 2
    sign = -1 if B < 0 else 1
    s = (triangle[0, 1] * triangle[2, 0] - triangle[0, 0] * triangle[2, 1] + (triangle[2, 1] - triangle[0, 1]) * p[
        0] + (triangle[0, 0] - triangle[2, 0]) * p[1]) * sign
    t = (triangle[0, 0] * triangle[1, 1] - triangle[0, 1] * triangle[1, 0] + (triangle[0, 1] - triangle[1, 1]) * p[
        0] + (triangle[1, 0] - triangle[0, 0]) * p[1]) * sign

    return s > 0 and t > 0 and (s + t) < 2 * B * sign


# Making mutation to 80% of the population
@njit(parallel=True)
def makeMutation(population):
    for i in range(size // 2):
        a = random.randint(0, 11)
        if (a <= 4):
            mutation_1(population[size // 2 + i])
        if (a >= 8):
            mutation_2(population[size // 2 + i])


# Iterations and showing the results
def main():
    population = create_population()
    for i in range(generations):
        print(i)
        population = fitness(population)
        if i%10 == 0:
            image = Image.fromarray(population[0], 'RGB')
            image.save('results/'+str(i)+'.jpg')
            # image.show()


main()
