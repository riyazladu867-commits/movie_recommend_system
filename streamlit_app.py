import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.rec-card {
    background: white; border-radius: 10px; padding: 1rem;
    border: 1px solid #e0e0e0; margin: 0.4rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,.05);
}
.main-title {
    font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(135deg, #1565C0, #AD1457);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center;
}
</style>""", unsafe_allow_html=True)

# ── Load all artifacts ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_all():
    with open("movie_rec_artifacts.pkl", "rb") as f:
        art = pickle.load(f)
    df  = pd.read_csv("movie_data_processed.csv")
    mat = art["tfidf_rec"].transform(df["tags"].fillna(""))
    return art, df, mat

art, df, tfidf_matrix = load_all()
best_model   = art["best_model"]
best_name    = art["best_name"]
tfidf_clf    = art["tfidf_clf"]
le           = art["label_encoder"]
results_df   = art["results_df"]
movie_titles = df["title"].fillna("Unknown").str.strip().values

# ── Auto-detect feature count for prediction ──────────────────────────
expected_features = best_model.n_features_in_
tfidf_features    = tfidf_clf.transform(["test"]).shape[1]
extra_features    = expected_features - tfidf_features
NUM_CANDIDATES    = ["vote_average","popularity","imdb_rating","rating_score"]
num_cols_used     = [c for c in NUM_CANDIDATES if c in df.columns][:extra_features]
medians           = {c: float(df[c].median()) for c in num_cols_used}

def build_features(tag_text, row=None):
    text_part = tfidf_clf.transform([str(tag_text)])
    if extra_features <= 0:
        return text_part
    num_vals = []
    for col in num_cols_used:
        if row is not None and col in df.columns:
            val = float(row[col]) if pd.notna(row.get(col)) else medians[col]
        else:
            val = medians[col]
        num_vals.append(val)
    while len(num_vals) < extra_features:
        num_vals.append(0.0)
    num_part = csr_matrix(
        np.array(num_vals[:extra_features], dtype=np.float64).reshape(1, -1))
    return hstack([text_part, num_part])

# ── Header ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Movie Recommendation System</div>',
            unsafe_allow_html=True)
st.caption(f"Best Model: **{best_name}** | "
           f"Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f} | "
           f"Dataset: {len(df):,} movies")
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    top_n        = st.slider("Number of Recommendations", 5, 20, 10)
    grade_filter = st.multiselect("Filter by Grade",
                                  ["High", "Medium", "Low"],
                                  default=["High", "Medium", "Low"])
    min_sim = st.slider("Min Similarity Score", 0.0, 0.3, 0.0, 0.01)

    st.divider()
    st.markdown("### 🏆 Model Results")
    st.dataframe(
        results_df[["Model","Test Accuracy","F1-Score","Precision","Recall"]]
        .style.format({c:"{:.4f}" for c in
                       ["Test Accuracy","F1-Score","Precision","Recall"]}),
        use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📊 Dataset Info")
    st.info(f"**{len(df):,}** movies loaded")
    gc_map = {"High":"🟢","Medium":"🟡","Low":"🔴"}
    for cls, cnt in df["rating_class"].value_counts().items():
        st.write(f"{gc_map.get(cls,'⚪')} **{cls}**: {cnt} ({cnt/len(df)*100:.0f}%)")

# ── Search bar ─────────────────────────────────────────────────────────
st.markdown("### 🔍 Search a Movie")
col_inp, col_btn = st.columns([4, 1])
with col_inp:
    movie_input = st.text_input("Movie title", label_visibility="collapsed",
                                placeholder="e.g. The Dark Knight, Inception, Avatar...")
with col_btn:
    search_btn = st.button("Search 🔍", use_container_width=True)

# Quick pick buttons
st.markdown("**✨ Quick picks:**")
qcols = st.columns(6)
quick = ["The Dark Knight","Inception","Avatar","Titanic","Interstellar","The Avengers"]
for qcol, qm in zip(qcols, quick):
    if qcol.button(qm, use_container_width=True):
        movie_input = qm
        search_btn  = True

# ── Recommendation logic ───────────────────────────────────────────────
if (search_btn or movie_input) and movie_input.strip():
    title_lower = movie_input.lower().strip()
    matches     = [i for i, t in enumerate(movie_titles)
                   if title_lower in str(t).lower()]

    if not matches:
        st.error(f"❌ Movie **'{movie_input}'** not found. Try a shorter or partial title.")
        st.info("💡 Example: type **'dark'** instead of 'The Dark Knight'")
    else:
        idx    = matches[0]
        ftitle = movie_titles[idx]
        frow   = df.iloc[idx]

        # Input movie info
        st.success(f"✅ Found: **{ftitle}**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("⭐ Rating",   f"{frow.get('rating_score', 0):.1f}")
        grade = frow.get("rating_class", "N/A")
        m2.metric("🏷️ Grade",   f"{gc_map.get(grade,'⚪')} {grade}")
        m3.metric("🎭 Genres",  str(frow.get("genres_clean",""))[:25] or "N/A")
        m4.metric("🎬 Director",str(frow.get("director_clean",""))[:22] or "N/A")

        # Compute similarities
        with st.spinner("Finding similar movies..."):
            sims        = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            sims[idx]   = 0
            top_idx     = sims.argsort()[::-1][:top_n * 4]

        gc = {"High":"#1B5E20","Medium":"#F57F17","Low":"#B71C1C"}
        ge = {"High":"🟢","Medium":"🟡","Low":"🔴"}

        recs = []
        for i in top_idx:
            if sims[i] < min_sim: continue
            row  = df.iloc[i]
            feat = build_features(str(row.get("tags","")), row)
            pred = le.inverse_transform(best_model.predict(feat))[0]
            if pred not in grade_filter: continue
            recs.append({
                "Title":      movie_titles[i],
                "Genres":     str(row.get("genres_clean",""))[:35],
                "Director":   str(row.get("director_clean",""))[:25],
                "Rating":     round(float(row.get("rating_score",0)), 1),
                "Grade":      pred,
                "Similarity": round(float(sims[i]), 4),
            })
            if len(recs) >= top_n: break

        if not recs:
            st.warning("No results match your filters. Try changing Grade filter or Min Similarity.")
        else:
            st.markdown(f"### 🎯 Top {len(recs)} Recommendations for *{ftitle}*")

            # Recommendation cards
            for rank, r in enumerate(recs, 1):
                c   = gc.get(r["Grade"], "#1565C0")
                e   = ge.get(r["Grade"], "⚪")
                sim_pct = int(r["Similarity"] * 100)
                bar = "█" * (sim_pct // 5) + "░" * (20 - sim_pct // 5)
                st.markdown(f"""
                <div class="rec-card">
                    <span style="color:#aaa;font-size:0.85rem">#{rank}</span>&nbsp;
                    <b style="font-size:1.05rem">{r['Title']}</b> &nbsp;
                    <span style="color:{c};font-weight:bold">{e} {r['Grade']}</span>
                    &nbsp;|&nbsp; ⭐ <b>{r['Rating']}</b>
                    &nbsp;|&nbsp; 🎭 {r['Genres'] or 'N/A'}
                    &nbsp;|&nbsp; 🎬 {r['Director'] or 'N/A'}
                    <br>
                    <small style="color:#888">
                        Similarity: <b style="color:{c}">{r['Similarity']:.4f}</b>
                        &nbsp; <code style="color:{c}">{bar}</code> {sim_pct}%
                    </small>
                </div>""", unsafe_allow_html=True)

            st.divider()

            # ── Bar chart using st.bar_chart (no matplotlib needed)
            st.markdown("#### 📊 Similarity Scores")
            rdf = pd.DataFrame(recs)
            chart_data = rdf.set_index("Title")[["Similarity"]].sort_values("Similarity")
            st.bar_chart(chart_data, height=400)

            # ── Full table
            st.markdown("#### 📋 Full Results Table")
            st.dataframe(
                rdf[["Title","Rating","Grade","Similarity","Genres","Director"]],
                use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"<center><small>🎬 Movie Recommendation System · "
    f"Best Model: <b>{best_name}</b> · "
    f"Acc: {results_df.iloc[0]['Test Accuracy']:.4f} · "
    f"Built with Streamlit + Scikit-learn</small></center>",
    unsafe_allow_html=True)
