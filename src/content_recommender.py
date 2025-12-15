import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .db import fetch_data

class ContentBasedRecommender:
    def __init__(self):
        self.df_movies = None
        self.df_prefs = None
        self.tfidf_matrix = None
        self.sim_matrix = None
        self.id_to_index = None
        self.tfidf = None
        
        # Stopwords, palabras que no aportan valor al texto, a eliminar para no enuciar en el análisis
        self.stopwords_es = [
            "la", "las", "el", "los", "de", "del", "y", "o", "un", "una", "unos", "unas",
            "que", "con", "a", "en", "por", "para", "su", "sus", "al", "es"
        ]

    def load_and_process_data(self):
        """
        Carga datos de Supabase, procesa el texto y genera la matriz TF-IDF y de similitud.
        """
        # 1. Obtener datos de la BD (Supabase)
        prefs_data, movies_data = fetch_data()
        
        self.df_prefs = pd.DataFrame(prefs_data)
        self.df_movies = pd.DataFrame(movies_data)

        if self.df_prefs.empty or self.df_movies.empty:
            print("No hay suficientes datos en la BD para el recomendador de contenido.")
            return

        print(f"Datos de contenido cargados: {len(self.df_movies)} películas.")

        # 2. Preprocesamiento de texto
        # Aseguramos que existan las columnas, si no, rellenamos con vacío
        if 'genero' not in self.df_movies.columns:
            self.df_movies['genero'] = ''
        if 'descripción' not in self.df_movies.columns:
            # Intentamos buscar 'descripcion' sin tilde por si acaso
            if 'descripcion' in self.df_movies.columns:
                self.df_movies['descripción'] = self.df_movies['descripcion']
            else:
                self.df_movies['descripción'] = ''

        # Crear texto combinado: Genero + Descripción, así obtenemos un solo bloque de texto que represente la esencia de la película
        self.df_movies["texto"] = (
            self.df_movies["genero"].fillna('') + " " + 
            self.df_movies["descripción"].fillna('')
        )

        # 3. Vectorización TF-IDF ( si una palabra aparece en muchos documentos, se reduce su peso, si aparece poco significa que es muy importante)
        self.tfidf = TfidfVectorizer(stop_words=self.stopwords_es)

        # Transformamos el texto en una matriz de TF-IDF donde las columnas son las palabras y las filas son las películas
        self.tfidf_matrix = self.tfidf.fit_transform(self.df_movies["texto"])

        # 4. Matriz de Similitud (Coseno)
        # Calculamos la similitud entre todas las películas
        self.sim_matrix = cosine_similarity(self.tfidf_matrix)

        # 5. Mapeo de ID de película a índice del DataFrame
        # Asumimos que 'id_pelicula' es la PK
        self.id_to_index = {
            int(row["id_pelicula"]): i 
            for i, row in self.df_movies.iterrows()
        }
        
        print("✅ Matriz de similitud de contenido calculada.")

    def get_available_genres(self):
        """Retorna una lista de géneros únicos disponibles en la BD."""
        if self.df_movies is None or 'genero' not in self.df_movies.columns:
            return []
            
        generos_set = set()
        for lista_generos in self.df_movies["genero"].dropna():
            partes = lista_generos.split("/") 
            for g in partes:
                generos_set.add(g.strip())
        
        return sorted(list(generos_set))

    def recommend_for_new_user(self, user_genres, top_n=5):
        """
        Recomendación para Cold Start basada en géneros ingresados por el usuario.
        user_genres: lista de strings o string con géneros.
        """
        if self.tfidf is None:
            return []

        # Si pasan una lista, la unimos en un solo string
        if isinstance(user_genres, list):
            texto_usuario = " ".join(user_genres)
        else:
            texto_usuario = str(user_genres)

        # Vector TF-IDF del usuario nuevo
        user_vec = self.tfidf.transform([texto_usuario])

        # Similaridad usuario vs todas las películas
        sim_scores = cosine_similarity(user_vec, self.tfidf_matrix)[0]

        # Añadimos puntaje temporal al DF para ordenar
        # Trabajamos sobre una copia para no afectar el original
        df_temp = self.df_movies.copy()
        df_temp["similarity_newuser"] = sim_scores

        # Top N
        recs = df_temp.sort_values("similarity_newuser", ascending=False).head(top_n)

        results = []
        for _, row in recs.iterrows():
            results.append({
                "id_pelicula": row["id_pelicula"],
                "titulo": row["titulo"],
                "score": round(row["similarity_newuser"], 4)
            })
            
        return results

    def recommend_for_existing_user(self, user_id, top_n=5):
        """
        Recomendación basada en el historial del usuario (Content-Based).
        """
        if self.df_prefs is None or self.sim_matrix is None:
            return []

        # Filtrar historial del usuario
        vistas = self.df_prefs[self.df_prefs["id_usuario"] == user_id]

        if vistas.empty:
            return [] # No tiene historial, debería usarse recommend_for_new_user

        scores = {}

        for _, row in vistas.iterrows():
            id_peli = int(row["id_pelicula"])
            punt = float(row["puntaje"])

            if id_peli not in self.id_to_index:
                continue

            idx = self.id_to_index[id_peli]
            
            # Similitud de esta película con TODAS las demás
            sim_scores_row = self.sim_matrix[idx]

            for i, score in enumerate(sim_scores_row):
                # Obtenemos el ID de la película destino
                id_recom = int(self.df_movies.iloc[i]["id_pelicula"])

                # Evitar recomendar lo ya visto
                if id_recom in vistas["id_pelicula"].values:
                    continue

                # Acumular puntaje: Similitud * Rating que dio el usuario
                scores[id_recom] = scores.get(id_recom, 0.0) + (score * punt)

        # Ordenar por mayor puntaje acumulado
        ordenado = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for peli_id, score in ordenado[:top_n]:
            # Buscar info de la peli
            if peli_id in self.id_to_index:
                idx_peli = self.id_to_index[peli_id]
                titulo = self.df_movies.iloc[idx_peli]["titulo"]
                results.append({
                    "id_pelicula": peli_id,
                    "titulo": titulo,
                    "score": round(score, 4)
                })

        return results
