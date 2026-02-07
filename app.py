import streamlit as st
import pandas as pd
import time
import random
import itertools
import matplotlib.pyplot as plt
import tsp_base

# -----------------------------
# Configuración
# -----------------------------
st.set_page_config(page_title="TSP Matrix", layout="wide")
st.title("🧬 TSP – Evolución con PMX (8! exacto)")
st.markdown("Selección sándwich + prevención de colapso genético")

# -----------------------------
# Parámetros FIJOS (por 8!)
# -----------------------------
NUM_CIUDADES = 8
TAM_POB = 10

with st.sidebar:
    st.info("Experimento fijado en 8 ciudades (8! = 40,320 rutas)")
    GENERACIONES = st.slider("Generaciones", 10, 150, 60)
    VELOCIDAD = st.slider("Velocidad (ms)", 50, 500, 150) / 1000
    MUTACION_PROB = 0.25

# -----------------------------
# Estado
# -----------------------------
if 'matriz_costos' not in st.session_state:
    st.session_state.ciudades = tsp_base.crear_ciudades(NUM_CIUDADES)
    st.session_state.matriz_costos = tsp_base.crear_matriz_costos(NUM_CIUDADES)

ciudades = st.session_state.ciudades
matriz_costos = st.session_state.matriz_costos

# -----------------------------
# Mostrar matriz
# -----------------------------
with st.expander("📊 Ver Matriz de Costos"):
    df = pd.DataFrame(
        matriz_costos,
        columns=[f"C{i}" for i in ciudades],
        index=[f"C{i}" for i in ciudades]
    )
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
# Óptimo global (8!)
# -----------------------------
def calcular_optimo_global():
    mejor = None
    mejor_costo = float('inf')

    for perm in itertools.permutations(ciudades):
        costo = calcular_fitness(list(perm))
        if costo < mejor_costo:
            mejor_costo = costo
            mejor = list(perm)

    return mejor, mejor_costo

# -----------------------------
# PMX
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
# Población inicial (sin duplicados)
# -----------------------------
def crear_poblacion():
    pob = []
    while len(pob) < TAM_POB:
        r = ciudades[:]
        random.shuffle(r)
        if r not in pob:
            pob.append(r)
    return pob

# -----------------------------
# EJECUCIÓN
# -----------------------------
if st.button("🚀 INICIAR SIMULACIÓN"):

    optimo_global, costo_optimo = calcular_optimo_global()
    poblacion = crear_poblacion()

    col_tabla, col_graf = st.columns([2.3, 1])
    placeholder = col_tabla.empty()
    metric = col_graf.empty()
    grafica = col_graf.empty()

    mejor_costo = float('inf')
    mejor_ruta = None
    hist_ga = []
    hist_opt = []

    for gen in range(GENERACIONES):

        evaluados = [(ind, calcular_fitness(ind)) for ind in poblacion]
        evaluados.sort(key=lambda x: x[1])

        if evaluados[0][1] < mejor_costo:
            mejor_costo = evaluados[0][1]
            mejor_ruta = evaluados[0][0]

        hist_ga.append(mejor_costo)
        hist_opt.append(costo_optimo)

        datos = [[f"C{i}" for i in ind] + [cost] for ind, cost in evaluados]
        cols = [f"P{i+1}" for i in range(NUM_CIUDADES)] + ["COSTO"]
        df_view = pd.DataFrame(datos, columns=cols)

        def estilo(row):
            if row["COSTO"] == mejor_costo:
                return ['background-color:#00ff00; color:black; font-weight:bold'] * len(row)
            return [''] * len(row)

        placeholder.dataframe(
            df_view.style.apply(estilo, axis=1),
            use_container_width=True
        )

        metric.metric(
            f"Generación {gen+1}",
            f"Mejor GA: {mejor_costo} | Óptimo: {costo_optimo}"
        )

        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.plot(hist_ga, label="GA", linewidth=2)
        ax.plot(hist_opt, label="Óptimo", linestyle="--")
        ax.set_xlabel("Generación")
        ax.set_ylabel("Costo")
        ax.legend()
        ax.grid(True)
        grafica.pyplot(fig)
        plt.close(fig)

        nueva = [evaluados[0][0]]
        i, j = 0, len(evaluados) - 1

        while i < j and len(nueva) < TAM_POB:
            p1 = evaluados[i][0]
            p2 = evaluados[j][0]

            h1, h2 = pmx(p1, p2)
            h1 = mutar(h1)
            h2 = mutar(h2)

            if h1 not in nueva:
                nueva.append(h1)
            if len(nueva) < TAM_POB and h2 not in nueva:
                nueva.append(h2)

            i += 1
            j -= 1

        while len(nueva) < TAM_POB:
            r = ciudades[:]
            random.shuffle(r)
            if r not in nueva:
                nueva.append(r)

        poblacion = nueva
        time.sleep(VELOCIDAD)

    st.success("Optimización completada")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧬 Mejor solución del GA")
        df_ga = pd.DataFrame([[f"C{i}" for i in mejor_ruta]],
                             columns=[f"P{i+1}" for i in range(NUM_CIUDADES)])
        st.dataframe(df_ga, use_container_width=True)
        st.write(f"**Costo:** {mejor_costo}")

    with col2:
        st.subheader("🌍 Óptimo Global")
        df_opt = pd.DataFrame([[f"C{i}" for i in optimo_global]],
                              columns=[f"P{i+1}" for i in range(NUM_CIUDADES)])
        st.dataframe(df_opt, use_container_width=True)
        st.write(f"**Costo:** {costo_optimo}")

