import streamlit as st
import pandas as pd
import time
import random
import tsp_base

# -----------------------------
# Configuración
# -----------------------------
st.set_page_config(page_title="TSP Matrix", layout="wide")
st.title("🧬 TSP – Evolución con PMX (Efecto Matrix)")
st.markdown("Permutaciones + PMX de Michalewicz")

# -----------------------------
# Parámetros
# -----------------------------
with st.sidebar:
    NUM_CIUDADES = st.slider("Cantidad de Ciudades", 5, 15, 6)
    GENERACIONES = st.slider("Generaciones", 10, 100, 40)
    VELOCIDAD = st.slider("Velocidad (ms)", 50, 500, 150) / 1000
    MUTACION_PROB = 0.2

# -----------------------------
# Estado
# -----------------------------
if 'matriz_costos' not in st.session_state or st.session_state.n != NUM_CIUDADES:
    st.session_state.ciudades = tsp_base.crear_ciudades(NUM_CIUDADES)
    st.session_state.matriz_costos = tsp_base.crear_matriz_costos(NUM_CIUDADES)
    st.session_state.n = NUM_CIUDADES

ciudades = st.session_state.ciudades
matriz_costos = st.session_state.matriz_costos

# -----------------------------
# Mostrar matriz
# -----------------------------
with st.expander("Ver Matriz de Costos"):
    df = pd.DataFrame(matriz_costos,
        columns=[f"C{i}" for i in ciudades],
        index=[f"C{i}" for i in ciudades])
    st.dataframe(df)

# -----------------------------
# Fitness
# -----------------------------
def calcular_fitness(ruta):
    costo = 0
    for i in range(len(ruta)):
        a = ruta[i]
        b = ruta[(i + 1) % len(ruta)]
        costo += matriz_costos[a][b]
    return costo

# -----------------------------
# PMX – Michalewicz
# -----------------------------
def pmx(p1, p2):
    n = len(p1)
    c1, c2 = [-1]*n, [-1]*n
    a, b = sorted(random.sample(range(n), 2))

    c1[a:b] = p1[a:b]
    c2[a:b] = p2[a:b]

    def completar(hijo, padre):
        for i in range(n):
            if hijo[i] == -1:
                val = padre[i]
                while val in hijo:
                    val = padre[hijo.index(val)]
                hijo[i] = val

    completar(c1, p2)
    completar(c2, p1)
    return c1, c2

# -----------------------------
# Mutación
# -----------------------------
def mutar(ruta):
    r = ruta[:]
    if random.random() < MUTACION_PROB:
        i, j = random.sample(range(len(r)), 2)
        r[i], r[j] = r[j], r[i]
    return r

# -----------------------------
# Población inicial
# -----------------------------
def crear_poblacion(tam):
    pob = []
    for _ in range(tam):
        r = ciudades[:]
        random.shuffle(r)
        pob.append(r)
    return pob

# -----------------------------
# EJECUCIÓN
# -----------------------------
if st.button("🚀 INICIAR SIMULACIÓN"):

    poblacion = crear_poblacion(10)
    placeholder = st.empty()
    metric = st.empty()

    mejor_global = None
    mejor_costo = float('inf')

    for gen in range(GENERACIONES):

        evaluados = [(ind, calcular_fitness(ind)) for ind in poblacion]
        evaluados.sort(key=lambda x: x[1])

        if evaluados[0][1] < mejor_costo:
            mejor_costo = evaluados[0][1]
            mejor_global = evaluados[0][0]

        # ---------------- MATRIX VIEW ----------------
        datos = []
        for ind, cost in evaluados:
            datos.append([f"C{i}" for i in ind] + [cost])

        cols = [f"P{i+1}" for i in range(NUM_CIUDADES)] + ["COSTO"]
        df_view = pd.DataFrame(datos, columns=cols)

        def estilo(row):
            if row["COSTO"] == mejor_costo:
                return ['background-color:#00ff00; color:black; font-weight:bold']*len(row)
            return ['']*len(row)

        placeholder.dataframe(df_view.style.apply(estilo, axis=1), use_container_width=True)
        metric.metric(f"Generación {gen+1}", f"Mejor costo: {mejor_costo}")

        # ---------------- EVOLUCIÓN ----------------
        nueva = []
        nueva.append(evaluados[0][0])  # elitismo

        while len(nueva) < len(poblacion):
            p1 = evaluados[0][0]
            p2 = random.choice(evaluados)[0]
            h1, h2 = pmx(p1, p2)
            nueva.append(mutar(h1))
            if len(nueva) < len(poblacion):
                nueva.append(mutar(h2))

        poblacion = nueva
        time.sleep(VELOCIDAD)

    st.success("Optimización completada")
    st.write("Mejor ruta encontrada:")
    st.write([f"C{i}" for i in mejor_global])
    st.write(f"Costo total: {mejor_costo}")
