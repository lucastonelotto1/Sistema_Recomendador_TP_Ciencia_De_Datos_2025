# Sistema Recomendador de Películas

**TP Integrador - Ciencia de Datos 2025**

Sistema de recomendación híbrido que combina **Filtrado Colaborativo** y **Filtrado Basado en Contenido** para sugerir películas a usuarios, implementado con FastAPI y conectado a una base de datos Supabase (PostgreSQL).

---

## Tabla de contenido

1. [¿Qué hace el sistema?](#-qué-hace-el-sistema)
2. [Estrategia de Switching](#-estrategia-de-switching)
3. [Métodos de Recomendación](#-métodos-de-recomendación)
   - [Filtrado Colaborativo (Item-Based)](#1-filtrado-colaborativo-item-based)
   - [Filtrado Basado en Contenido](#2-filtrado-basado-en-contenido)
4. [Cálculo de Similitud](#-cálculo-de-similitud)
5. [Instalación y Configuración](#-instalación-y-configuración)
6. [Endpoints de la API](#-endpoints-de-la-api)
7. [Frontend](#-frontend)

---

## ¿Qué hace el sistema?

El sistema recomienda películas a usuarios basándose en:

- Su historial de calificaciones (si tiene suficientes interacciones)
- El contenido de las películas (género, descripción) cuando el usuario es nuevo o tiene pocas interacciones

El objetivo es resolver el problema del Cold Start (usuarios nuevos sin historial) mediante una estrategia de switching que selecciona automáticamente el mejor método según la situación.

---

## Estrategia de Switching

El sistema implementa un switching híbrido que decide qué algoritmo usar según la cantidad de interacciones del usuario:

| Interacciones | Método Usado | Descripción |
|---------------|--------------|-------------|
| **≥ 6** | Filtrado Colaborativo (Item-Based) | Usuario con historial suficiente. |
| **1 - 5** | Content-Based (por historial) | Usuario con poco historial. Se recomienda basándose en las películas que ya vio. |
| **0** | Content-Based (por géneros) | Usuario totalmente nuevo. Se le piden sus géneros favoritos y se recomienda según eso. |

### ¿Por qué el umbal?

Es un umbral seleccionado para el caso de estudio, donde los usuarios suelen realizar usualmente 7 compras, con este umbral nos acercamos a su límite, si comenzamos a hacer mejores recomendaciones podríamos incentivar la compra antes de que finalice. Con menos de 6 calificaciones, el filtrado colaborativo no tiene suficiente información para encontrar patrones confiables, por lo que el sistema basado en contenido es más efectivo.

---

## Métodos de Recomendación

### 1. Filtrado Colaborativo (Item-Based)

**Archivo:** `src/recommender.py`

Este método recomienda películas basándose en la similitud entre ítems (películas), calculada a partir de cómo los usuarios las han calificado.

#### Proceso:

1. **Construcción de la Matriz Usuario-Película**
   ```
   Matriz Pivote (filas: usuarios, columnas: películas, valores: puntajes)
   
           Película1  Película2  Película3
   User1      5          3          0
   User2      4          0          2
   User3      0          5          4
   ```

2. **Cálculo de Similitud Item-Item**
   - Se transpone la matriz para tener películas como filas
   - Se calcula la similitud coseno entre cada par de películas
   - Resultado: matriz de similitud NxN (N = cantidad de películas)

3. **Generación de Recomendaciones**
   - Para cada película que el usuario calificó positivamente (puntaje > 2):
     - Se buscan películas similares
     - Se acumula un score: `similitud × puntaje_usuario`
   - Se filtran las películas ya vistas
   - Se ordenan por score descendente

#### Fórmula del Score:
```
Score(película_candidata) = Σ (similitud(película_vista, película_candidata) × puntaje_dado)
```

---

### 2. Filtrado Basado en Contenido

**Archivo:** `src/content_recommender.py`

Este método recomienda películas basándose en las características del contenido (género y descripción), sin depender de otros usuarios.

#### Proceso:

1. **Preprocesamiento de Texto**
   - Se concatena `género + descripción` de cada película
   - Se eliminan stopwords en español

2. **Vectorización TF-IDF**
   ```
   TF-IDF = Term Frequency × Inverse Document Frequency
   ```
   - Convierte el texto de cada película en un vector numérico
   - Palabras frecuentes en una película pero raras en el corpus tienen mayor peso

3. **Matriz de Similitud**
   - Se calcula similitud coseno entre todos los vectores TF-IDF
   - Resultado: matriz de similitud película×película

#### Dos modos de operación:

**a) Usuario con historial (1-5 interacciones):**
```python
def recommend_for_existing_user(user_id, top_n=5):
    # Para cada película vista por el usuario:
    #   Score(candidata) += similitud_contenido(vista, candidata) × puntaje
    # Retorna top_n ordenadas por score
```

**b) Usuario nuevo (0 interacciones):**
```python
def recommend_for_new_user(user_genres, top_n=5):
    # 1. Convierte los géneros del usuario en vector TF-IDF
    # 2. Calcula similitud con todas las películas
    # 3. Retorna las top_n más similares
```

---

## Cálculo de Similitud

Ambos métodos usan similitud coseno para medir qué tan parecidos son dos vectores:

### Fórmula:

$$
\text{similitud}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|} = \frac{\sum_{i=1}^{n} A_i \times B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

### Interpretación:

| Valor | Significado |
|-------|-------------|
| **1.0** | Vectores idénticos (máxima similitud) |
| **0.0** | Vectores ortogonales (sin relación) |
| **-1.0** | Vectores opuestos (en teoría, pero con TF-IDF siempre es ≥ 0) |

### En el Filtrado Colaborativo:
- Los vectores representan patrones de calificación de cada película
- Dos películas son similares si los mismos usuarios las calificaron de forma parecida

### En el Filtrado por Contenido:
- Los vectores representan características textuales (TF-IDF)
- Dos películas son similares si comparten género y palabras en la descripción

---

## Instalación y Configuración

### Requisitos

- Python 3.8+
- Cuenta en Supabase con las tablas `USUARIO`, `PELICULA`, `PREFERENCIA`

### Pasos

1. **Clonar/descargar el proyecto**

2. **Crear y activar el entorno virtual**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**
   ```powershell
   copy .env.example .env
   ```
   Editar `.env` con tus credenciales de Supabase:
   ```
   SUPABASE_URL=https://tu-proyecto.supabase.co
   SUPABASE_KEY=tu-api-key
   ```

5. **Ejecutar el servidor**
   ```powershell
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Abrir en el navegador**
   ```
   http://127.0.0.1:8000/
   ```

### Dependencias principales

| Paquete | Uso |
|---------|-----|
| `fastapi` | Framework web para la API |
| `uvicorn` | Servidor ASGI |
| `supabase` | Cliente para conectar con la BD |
| `pandas` | Manipulación de datos y matrices |
| `scikit-learn` | TF-IDF y similitud coseno |
| `python-dotenv` | Variables de entorno |

---

## Endpoints de la API

### Usuarios

#### `POST /user`
Crea un nuevo usuario.

**Body:**
```json
{
  "username": "nombreDeEjemplo"
}
```

**Respuesta (201):**
```json
{
  "id": 15,
  "username": "nombreDeEjemplo",
  "attributes": null
}
```

---

#### `GET /user/{userId}`
Obtiene información de un usuario.

**Respuesta (200):**
```json
{
  "id": 15,
  "username": "nombreDeEjemplo",
  "attributes": null
}
```

---

### Recomendaciones

#### `GET /user/{userId}/recommend`
Obtiene recomendaciones para un usuario.

**Parámetros Query:**
| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `n` | int | 5 | Cantidad de recomendaciones (1-50) |
| `generos` | string | null | Géneros favoritos separados por coma (solo para usuarios nuevos) |

**Ejemplo - Usuario con historial:**
```
GET /user/22/recommend?n=5
```

**Respuesta (200):**
```json
{
  "user_id": 22,
  "n_interacciones": 8,
  "metodo_usado": "colaborativo",
  "recomendaciones": [
    {"id_pelicula": 15, "titulo": "Star Wars: Una nueva esperanza", "score": 15.193},
    {"id_pelicula": 23, "titulo": "Terminator 2: El juicio final", "score": 10.1075}
  ]
}
```

**Ejemplo - Usuario nuevo (sin historial):**
```
GET /user/999/recommend?n=5&generos=Accion,Ciencia%20Ficcion
```

**Respuesta (200):**
```json
{
  "user_id": 999,
  "n_interacciones": 0,
  "metodo_usado": "contenido_generos",
  "recomendaciones": [
    {"id_pelicula": 12, "titulo": "Terminator 2", "score": 0.8765}
  ]
}
```

**Si el usuario es nuevo y no pasa géneros (400):**
```json
{
  "detail": {
    "message": "Usuario nuevo sin historial. Debe proporcionar géneros preferidos.",
    "generos_disponibles": ["Accion", "Aventura", "Comedia", "Drama", ...]
  }
}
```

---

#### `GET /generos`
Lista todos los géneros disponibles.

**Respuesta (200):**
```json
{
  "generos": ["Accion", "Aventura", "Ciencia Ficcion", "Comedia", "Drama", ...]
}
```

---

### Frontend

#### `GET /`
Sirve la interfaz web del sistema.

---

## Frontend

El sistema incluye una interfaz web simple accesible en `http://127.0.0.1:8000/` con:

- **Crear usuario:** Ingresá un username y se crea en la BD
- **Buscar usuario:** Consultá información por ID
- **Recomendar:** 
  - Si el usuario tiene historial → muestra recomendaciones directamente
  - Si es usuario nuevo → aparece un selector de géneros interactivo
- **Modo oscuro:** Toggle para cambiar el tema

---

## Estructura del Proyecto

```
TPIntegrador/
├── main.py                 # API FastAPI + endpoints
├── requirements.txt        # Dependencias
├── .env.example            # Template de variables de entorno
├── README.md               
├── src/
│   ├── __init__.py
│   ├── db.py               # Conexión a Supabase
│   ├── recommender.py      # Filtrado Colaborativo (Item-Based)
│   └── content_recommender.py  # Filtrado por Contenido
└── static/
    └── index.html          # Frontend web
```

---

## Autores

- Bogado Magalí
- Dodera Sofía
- Gomez Noelía
- Tonelotto Lucas
