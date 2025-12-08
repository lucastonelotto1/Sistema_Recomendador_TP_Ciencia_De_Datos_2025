import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .db import fetch_data

class ItemBasedRecommender:
    def __init__(self):
        self.df_merged = None
        self.item_similarity_df = None
        self.movies_info = None 
        self.user_movie_matrix = None 

    def load_and_process_data(self):
        """
        Carga, el join y la creación de la matriz.
        """
        # 1. Obtener datos crudos
        prefs_data, movies_data = fetch_data()
        
        df_prefs = pd.DataFrame(prefs_data)
        df_movies = pd.DataFrame(movies_data)

        if df_prefs.empty or df_movies.empty:
            print("No hay suficientes datos en la BD.")
            return

        self.movies_info = df_movies.set_index('id_pelicula')['titulo'].to_dict()

        # 3. Merge
        self.df_merged = pd.merge(df_prefs, df_movies, on='id_pelicula')
        print(f"📊 Datos cargados: {len(self.df_merged)} interacciones.")

        # 4. Crear la Tabla Pivote y GUARDARLA EN SELF
        self.user_movie_matrix = self.df_merged.pivot_table(
            index='id_usuario', 
            columns='id_pelicula', 
            values='puntaje'
        ).fillna(0) # Rellenamos con 0 lo que no se vio

        # 5. Calcular Similitud
        # Trasponemos para que las filas sean Películas
        item_user_matrix = self.user_movie_matrix.T
        
        similarity_matrix = cosine_similarity(item_user_matrix)

        self.item_similarity_df = pd.DataFrame(
            similarity_matrix,
            index=item_user_matrix.index,
            columns=item_user_matrix.index
        )
        print("Matriz de similitud calculada.")

    def mostrar_matrices_debug(self):
        """
        Muestra en consola un recorte de las matrices internas para inspección.
        """
        if self.user_movie_matrix is None:
            print("No hay datos cargados.")
            return

        print("\n" + "="*50)
        print("MODO DEBUG: VISUALIZANDO MATRICES")
        print("="*50)

        # 1. MATRIZ USUARIO-ITEM
        print(f"\n MATRIZ PIVOTE (Usuario x Item)")
        print(f"   (Filas: Usuarios, Columnas: Películas)")
        print(f"   Tamaño: {self.user_movie_matrix.shape}")
        print("-" * 30)
        # Mostramos solo las primeras 10 columnas para que quepa en pantalla
        print(self.user_movie_matrix.iloc[:10, :10].to_string()) 
        print("... (se muestran solo las primeras filas/columnas)")

        # 2. MATRIZ DE SIMILITUD
        print(f"\n MATRIZ DE SIMILITUD (Coseno Item-Item)")
        print(f"   (Filas: Películas, Columnas: Películas)")
        print(f"   Valores: 1.0 (Idénticas) a 0.0 (Distintas)")
        print("-" * 30)
        # Mostramos un recorte pequeño de 5x5
        print(self.item_similarity_df.iloc[:5, :5].to_string())
        print("="*50 + "\n")

    def get_interaction_count(self, user_id):
        """Devuelve la cantidad de películas que ha calificado el usuario."""
        if self.df_merged is None:
            return 0
        return len(self.df_merged[self.df_merged['id_usuario'] == user_id])

    def recommend_for_user(self, user_id, top_n=5):
        if self.item_similarity_df is None:
            return []

        user_history = self.df_merged[self.df_merged['id_usuario'] == user_id]
        if user_history.empty:
            return []

        candidate_items = {}

        for _, row in user_history.iterrows():
            movie_id = row['id_pelicula']
            rating = row['puntaje']
            
            if rating > 2:
                similar_movies = self.item_similarity_df[movie_id]
                for other_movie_id, similarity_score in similar_movies.items():
                    if other_movie_id == movie_id:
                        continue
                    candidate_items.setdefault(other_movie_id, 0)
                    candidate_items[other_movie_id] += similarity_score * rating

        watched_movies = set(user_history['id_pelicula'])
        final_recommendations = []
        
        for mid, score in candidate_items.items():
            if mid not in watched_movies:
                final_recommendations.append((mid, score))

        final_recommendations.sort(key=lambda x: x[1], reverse=True)

        results = []
        for mid, score in final_recommendations[:top_n]:
            title = self.movies_info.get(mid, str(mid))
            results.append({"id_pelicula": mid, "titulo": title, "score": round(score, 4)})

        return results