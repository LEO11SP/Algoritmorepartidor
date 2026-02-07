import numpy as np

def crear_ciudades(n):
    return list(range(n))

def crear_matriz_costos(n, min_costo=10, max_costo=100):
    matriz = np.random.randint(min_costo, max_costo + 1, size=(n, n))
    np.fill_diagonal(matriz, 0)
    return matriz
