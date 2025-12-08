import os
from supabase import create_client, Client
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

def get_supabase_client() -> Client:
    """Crea y devuelve el cliente de Supabase."""
    if not url or not key:
        raise ValueError("Faltan las credenciales SUPABASE_URL o SUPABASE_KEY en el .env")
    return create_client(url, key)

def fetch_data():
    """
    Trae los datos de PREFERENCIA y PELICULA.
    Retorna dos listas de diccionarios (data raw).
    """
    supabase = get_supabase_client()

    print("📡 Conectando a Supabase y descargando datos...")
    
    # 1. Traer Preferencias (el historial de ratings)
    # .select("*") trae todo. Si son muchos datos en el futuro, se paginaría.
    response_pref = supabase.table('PREFERENCIA').select("*").execute()
    
    # 2. Traer Películas (para saber los nombres)
    response_movies = supabase.table('PELICULA').select("*").execute()

    return response_pref.data, response_movies.data