# Tarea 4 — Redes bayesianas (IA 2027-I)

Enunciado: [assignment/hw4_redes_bayesianas.pdf](assignment/hw4_redes_bayesianas.pdf)

Regla clave: `pgmpy`/`bnlearn` **solo para verificar**. Todo lo demás (parser, d-separación,
inferencia, muestreo, aprendizaje) va hecho a mano con `numpy` + `pandas`.

## Estructura

```
hw4/
├── assignment/   enunciado
├── data/         child.bif, tratamiento.bif, child_datos_*.csv, child_prueba.csv,
│                 sachs.csv, sachs_referencia.txt   ← copiarlos aquí (aún no están)
├── bn/           la librería: módulos chicos, sin estado global, sin prints
├── scripts/      un script por inciso; solo llaman a bn/ y escriben en results/
├── results/      salidas (.txt/.csv/.png) que cita el reporte
└── report/       reporte (LaTeX o md) + críticas y ejercicios teóricos
```

Setup:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Idea central para bajar el acoplamiento

Una sola representación compartida y todo lo demás son funciones puras sobre ella:

- **Grafo** = `parents: dict[str, list[str]]`. Nada más. d-separación, VE, muestreo, BIC y SHD
  lo reciben así.
- **Red** = `states: dict[str, list[str]]` + `parents` + `cpts: dict[str, np.ndarray]`, con la
  CPT de `X` de forma `(*card(padres), card(X))` (última dimensión = la variable).
- **Datos** = matriz `int` `(n, d)` con códigos de estado (columnas en el orden de `states`).
  Se convierte el CSV una vez al cargar y ya nunca se vuelve a tocar texto.

Si respetas eso, cada módulo se puede probar solo y los scripts quedan de 20–40 líneas.

## Archivos mínimos sugeridos (`bn/`)

Firmas orientativas; cámbialas si se te ocurre algo mejor.

| Archivo | Qué contiene | Lo usa |
|---|---|---|
| `graph.py` | `topological_order(parents)`, `ancestors(parents, nodes)`, `is_acyclic(parents)`, `shd(true, learned)` | todo |
| `bif.py` | `load_bif(path) -> (states, parents, cpts)`; parser con regex, nada fancy | 2a, 2b, 2c |
| `dsep.py` | `d_separated(parents, x, y, z) -> (bool, camino_activo or None)` | 2a |
| `factor.py` | `Factor(vars, values)` + `multiply`, `sum_out`, `reduce(evidence)` | 2b, 3 |
| `inference.py` | `variable_elimination(net, query, evidence) -> (dist, p_e, stats)`; `stats` = orden, tamaño del factor más grande, tiempo | 2a.2, 2b |
| `sampling.py` | `forward_sample(net, n, rng)`, `rejection(...) -> (dist, n_aceptadas)`, `likelihood_weighting(...) -> (dist, n_ef)` | 2b.2–3 |
| `learning.py` | `encode(df, states)`, `fit_params(parents, data, states, alpha)`, `log_likelihood(net, data)`, `bic(parents, data, states)`, `hill_climb(data, states)` | 2c |

Scripts (`scripts/`): `p2a_dsep.py`, `p2b_inferencia.py`, `p2b_muestreo.py`,
`p2c_aprendizaje.py`, `p2c_sachs.py`, y uno `verificar_pgmpy.py` que compare tus números con
pgmpy (eso sí está permitido). Al final, un `resultados.ipynb` guía para el corrector.

## Orden sugerido (de lo que desbloquea más a lo que menos)

1. **`graph.py` + `bif.py`** — todo depende de esto. Verifica: CHILD tiene 20 variables y 25
   arcos, y cada CPT suma 1 en la última dimensión.
2. **2a d-separación** (8 pts, rápido). Busca "algoritmo Reachable / Bayes-ball" (Koller &
   Friedman, Alg. 3.1). Para devolver el camino activo, haz BFS sobre pares
   `(nodo, dirección)` y guarda el predecesor. Pruébalo primero en la red de 3a (B, I, M, G, J).
3. **`factor.py` + VE** (2b.1). Hazlo genérico con `np.einsum` o broadcasting; poda primero
   (ancestros de consulta ∪ evidencia) y elige el orden con una heurística greedy simple
   (min-degree o min-fill). Con VE ya resuelves también 2a.2 (`tratamiento.bif`).
4. **Muestreo** (2b.2–3). Vectoriza: genera las `N` muestras de una vez, nodo por nodo en orden
   topológico, indexando la CPT con los códigos de los padres. Así `N = 10^5` × 5 reps es
   instantáneo. `rng = np.random.default_rng(seed)` para reproducibilidad.
5. **Aprendizaje** (2c, 25 pts, el más pesado).
   - Parámetros = conteos por familia (`np.add.at` o `groupby`), `alpha=0` MLE, `alpha=1`
     Laplace. Ojo con `log(0)` en MLE al evaluar `child_prueba.csv`: decide y documenta qué
     hacer.
   - BIC es **descomponible**: cachea el score por `(hijo, frozenset(padres))`. Cada movimiento
     del hill climbing solo recalcula 1 o 2 familias. Sin caché es lento; con caché, segundos.
   - SHD en `graph.py`, para reutilizarlo con CHILD y Sachs.
6. **Teoría** (secciones 1 y 3) — se puede hacer en paralelo. Truco: escribe la red de 3a
   (B, I, M, G, J) como `.bif` en `data/` y verifica 3a(b)–(c) con tu propio VE.

## Dónde practicar PyTorch si sobra tiempo

Primero la versión numpy (rápida y simple); estas son extensiones opcionales, no reemplazos:

- **NOTEARS (pregunta 1b)** — el mejor lugar. Implementa la versión lineal con
  `torch.linalg.matrix_exp`, `h(W) = tr(e^{W∘W}) - d` y lagrangiano aumentado con autograd.
  Córrelo sobre `sachs.csv` (códigos 0/1/2 como si fueran continuos) y compáralo con tu hill
  climbing: te da material concreto para 1b.1 y 1b.2.
- **Parámetros por gradiente** — CPTs como `softmax(logits)`, maximiza la log-verosimilitud con
  Adam y comprueba que converge a los conteos MLE; agregar un prior Dirichlet ↔ Laplace.
- **Muestreo en GPU/batch** — portar `forward_sample` a `torch.multinomial` es casi directo.

## Pendientes antes de empezar

- [ ] Copiar los archivos de datos a `data/` (el enunciado los menciona pero no venían con el PDF).
- [ ] Crear el venv e instalar `requirements.txt`.
