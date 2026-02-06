# tsp_base.py
import numpy as np

def crear_ciudades(n):
    return list(range(n))

def crear_matriz_costos(n, min_costo=10, max_costo=100):
    # Matriz aleatoria NO simétrica
    matriz = np.random.randint(min_costo, max_costo + 1, size=(n, n))

    # La diagonal debe ser 0 (no cuesta ir a la misma ciudad)
    np.fill_diagonal(matriz, 0)

    return matriz
