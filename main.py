import os
import math
import random
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from supabase import create_client, Client
from postgrest.exceptions import APIError
from dotenv import load_dotenv

# Importar los recomendadores
from src.recommender import ItemBasedRecommender
from src.content_recommender import ContentBasedRecommender

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Sistema Recomendador - TP Ciencia de Datos")

# --- INICIALIZACIÓN DE RECOMENDADORES (al arrancar el servidor) ---
print("Inicializando Sistema de Recomendación...")

print("\n[1/2] Cargando Recomendador Colaborativo (Item-Based)...")
collab_rec = ItemBasedRecommender()
collab_rec.load_and_process_data()

print("\n[2/2] Cargando Recomendador de Contenido (Content-Based)...")
content_rec = ContentBasedRecommender()
content_rec.load_and_process_data()

print("\n Sistema de recomendación listo.\n")


class UserCreate(BaseModel):
    username: str
    attributes: Optional[Dict[str, Any]] = None


class User(BaseModel):
    id: int
    username: str
    attributes: Optional[Dict[str, Any]] = None


class RecommendationItem(BaseModel):
    id_pelicula: int
    titulo: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    n_interacciones: int
    metodo_usado: str
    recomendaciones: List[RecommendationItem]


@app.post("/user", response_model=User, status_code=201)
def create_user(user: UserCreate):
    # Insert only the fields we expect; Supabase will generate the id
    payload = {"username": user.username}
    if user.attributes is not None:
        payload["attributes"] = user.attributes

    # Insert the row; the client returns the created row in `response.data`
    try:
        response = supabase.table("USUARIO").insert(payload).execute()
    except APIError as e:
        # Handle database constraint errors (e.g., duplicate key)
        msg = getattr(e, "args", [str(e)])[0]
        # Return a 409 Conflict for duplicate key errors
        if isinstance(msg, dict) and msg.get("code") == "23505":
            raise HTTPException(status_code=409, detail=msg.get("message"))
        raise HTTPException(status_code=500, detail=str(msg))

    data = getattr(response, "data", None)
    if not data:
        err = getattr(response, "error", None)
        detail = "No se pudo crear el usuario en la base de datos"
        if err:
            detail += f": {err}"
        raise HTTPException(status_code=500, detail=detail)

    row = data[0]
    # Try common id column names used in the schema
    created_id = row.get("id_usuario") or row.get("id")
    if created_id is None:
        raise HTTPException(status_code=500, detail="El registro fue creado pero no se devolvió el id")

    return User(id=created_id, username=row.get("username"), attributes=row.get("attributes"))


@app.get("/user/{userId}", response_model=User)
def get_user(userId: int):
    response = supabase.table("USUARIO").select("*").eq("id_usuario", userId).execute()
    data = getattr(response, "data", None)
    if not data or len(data) == 0:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")

    row = data[0]
    return User(id=row.get("id_usuario") or row.get("id"), username=row.get("username"), attributes=row.get("attributes"))


@app.get("/user/{userId}/recommend", response_model=RecommendationResponse)
def recommend(
    userId: int,
    n: int = Query(default=5, ge=1, le=50, description="Cantidad de recomendaciones"),
    generos: Optional[str] = Query(default=None, description="Géneros preferidos (separados por coma) para usuarios nuevos")
):
    """
    Endpoint de recomendación con SWITCHING HÍBRIDO:
    - Si el usuario tiene >= 6 interacciones → Filtro Colaborativo (Item-Based)
    - Si tiene entre 1 y 5 interacciones → Content-Based (basado en historial)
    - Si tiene 0 interacciones → Content-Based (basado en géneros proporcionados)
    """
    # Verificar cuántas interacciones tiene el usuario
    n_interacciones = collab_rec.get_interaction_count(userId)
    
    recommendations = []
    metodo = ""

    # Calcular split 2/3 (recom) vs 1/3 (random)
    n_rec = math.ceil(n * (2/3))
    n_random = n - n_rec

    # --- 1. Obtener Recomendaciones del Sistema ---
    if n_interacciones >= 6:
        # Usuario con historial suficiente → Filtro Colaborativo
        metodo = "colaborativo"
        recommendations = collab_rec.recommend_for_user(userId, top_n=n_rec)
        
    elif n_interacciones > 0:
        # Usuario con poco historial → Content-Based usando su historial
        metodo = "contenido_historial"
        recommendations = content_rec.recommend_for_existing_user(userId, top_n=n_rec)
        
    else:
        # Usuario totalmente nuevo (0 interacciones) → Content-Based por géneros
        metodo = "contenido_generos"
        if generos:
            generos_lista = [g.strip() for g in generos.split(",")]
            recommendations = content_rec.recommend_for_new_user(generos_lista, top_n=n_rec)
        else:
            # Si no pasó géneros, devolver géneros disponibles como sugerencia
            available = content_rec.get_available_genres()
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Usuario nuevo sin historial. Debe proporcionar géneros preferidos en el parámetro 'generos'.",
                    "generos_disponibles": available[:20]
                }
            )

    # --- 2. Rellenar con Aleatorias (si corresponde) ---
    if n_random > 0:
        # Recuperar IDs de películas que YA vio el usuario para no recomendarlas
        # Usamos collab_rec.df_merged que tiene todo el historial
        watched_ids = set()
        if collab_rec.df_merged is not None:
             hist = collab_rec.df_merged[collab_rec.df_merged['id_usuario'] == userId]
             watched_ids = set(hist['id_pelicula'].unique())

        # También excluir las que ya recomendamos recién
        recom_ids = {r['id_pelicula'] for r in recommendations}
        
        excluded_ids = watched_ids.union(recom_ids)
        
        # Obtener pool de películas (usamos content_rec.df_movies que tiene 'titulo')
        if content_rec.df_movies is not None:
            # Filtrar las que no están en excluded
            candidate_movies = content_rec.df_movies[~content_rec.df_movies['id_pelicula'].isin(excluded_ids)]
            
            if not candidate_movies.empty:
                # Tomar muestra aleatoria
                n_take = min(n_random, len(candidate_movies))
                random_sample = candidate_movies.sample(n=n_take)
                
                for _, row in random_sample.iterrows():
                    recommendations.append({
                        "id_pelicula": row['id_pelicula'],
                        "titulo": row['titulo'],
                        "score": 0.0 # Score 0 para indicar que es aleatoria/fill
                    })
            
            metodo += f" + {n_random} aleatorias"

    return RecommendationResponse(
        user_id=userId,
        n_interacciones=n_interacciones,
        metodo_usado=metodo,
        recomendaciones=[RecommendationItem(**r) for r in recommendations]
    )


@app.get("/generos")
def get_generos():
    """Devuelve la lista de géneros disponibles en la base de datos."""
    return {"generos": content_rec.get_available_genres()}


@app.get("/", response_class=HTMLResponse)
def root():
    # Serve the simple frontend
    return FileResponse("static/index.html")
