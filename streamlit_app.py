# ----------------------------------------------------------------------------
# streamlit_app.py — Final: slicing + gráfico solo si embeddings están listos
# ----------------------------------------------------------------------------

import importlib, sys, os, re
try:
    import pysqlite3
    sys.modules["sqlite3"] = importlib.import_module("pysqlite3")
    sys.modules["sqlite3.dbapi2"] = sys.modules["sqlite3"]
except ImportError:
    pass

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ---------------------- Sidebar UI -------------------------------------------

st.sidebar.title("⚙️ Configuración")

vectorstore_path = st.sidebar.text_input("Ruta del vectorstore", "./chroma")

model_choice = "text-embedding-3-small"
target_dim   = 1024
st.sidebar.markdown("🧠 **Modelo:** `text-embedding-3-small`  \n📏 **Dimensión del embedding:** `1024`")

api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password")
show_pca = st.sidebar.checkbox("Mostrar visualización 3D (PCA)")
do_search = False

# ---------------------- Interfaz principal -----------------------------------

st.title("🔎 Buscador semántico de concursos")

query = st.text_input("Describe el servicio que buscas")
col_btn, col_k = st.columns([1, 1])
with col_k:
    top_k = st.number_input("Top‑K", min_value=1, max_value=20, value=5, step=1)
with col_btn:
    do_search = st.button("Buscar", use_container_width=True)

# -------------------- Embeddings con slicing manual --------------------------

def sliced_embedder_factory(api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key
    base_embedder = OpenAIEmbeddings(model=model_choice)

    class Wrapper:
        def embed_documents(self, texts):
            return [v[:target_dim] for v in base_embedder.embed_documents(texts)]

        def embed_query(self, text):
            return base_embedder.embed_query(text)[:target_dim]

    return Wrapper()

# ------------------------ Ejecución de búsqueda -------------------------------

if do_search:
    if not api_key.strip():
        st.warning("Por favor ingresa tu clave de API.")
        st.stop()

    if not query.strip():
        st.warning("Por favor ingresa una consulta.")
        st.stop()

    try:
        embedder = sliced_embedder_factory(api_key)
        vs = Chroma(persist_directory=vectorstore_path, embedding_function=embedder)

        try:
            n_docs = vs._collection.count()
        except Exception:
            n_docs = "?"
        st.sidebar.markdown(f"🗂️ **Documentos en colección: {n_docs}**")

        docs_scores = vs.similarity_search_with_relevance_scores(query, k=top_k)

        if not docs_scores:
            st.info("No se encontraron documentos relevantes.")
            st.stop()

        rows = []
        valid_ids = []
        for i, (doc, score) in enumerate(docs_scores):
            m = doc.metadata
            doc_id = m.get("id", m.get("codigo", str(i)))
            valid_ids.append(doc_id)
            rows.append({
                "Código":       m.get("codigo", "-"),
                "Entidad":      m.get("entidad", "-"),
                "Estado":       m.get("estado", "-"),
                "Publicación":  m.get("publicacion", "-"),
                "Similitud":    round(score, 4),
                "Link":         m.get("link", "-"),
                "Descripción":  (doc.page_content[:180] + "…") if len(doc.page_content) > 180 else doc.page_content
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # -------------------- Visualización 3D opcional ----------------------
        if show_pca:
            data = vs._collection.get(ids=valid_ids)
            if "embeddings" not in data or not data["embeddings"]:
                data = vs._collection.get(ids=valid_ids, include=["embeddings"])

            emb_list = data.get("embeddings", [])
            emb_valid = [e for e in emb_list if isinstance(e, list) and len(e) == target_dim]

            if len(emb_valid) >= 3:
                emb = np.array(emb_valid)
                pca = PCA(n_components=3)
                pts_3d = pca.fit_transform(emb)
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                    x=pts_3d[:, 0], y=pts_3d[:, 1], z=pts_3d[:, 2],
                    mode='markers+text',
                    text=[f"{r['Código']}<br>{r['Similitud']:.3f}" for r in rows[:len(pts_3d)]],
                    marker=dict(size=5)
                ))
                fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se pudo generar el gráfico (menos de 3 embeddings válidos).")

    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
