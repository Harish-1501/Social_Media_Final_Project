# app.py
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
from pathlib import Path

# -------------------------
# Model defs (only to extract item embeddings)
# -------------------------
class MF(nn.Module):
    def __init__(self, num_users, num_items, k=64):
        super().__init__()
        self.user_f = nn.Embedding(num_users, k)
        self.item_f = nn.Embedding(num_items, k)
        self.user_b = nn.Embedding(num_users, 1)
        self.item_b = nn.Embedding(num_items, 1)
        self.global_b = nn.Parameter(torch.zeros(1))

    def forward(self, u, i):
        pu = self.user_f(u)
        qi = self.item_f(i)
        bu = self.user_b(u).squeeze(-1)
        bi = self.item_b(i).squeeze(-1)
        dot = (pu * qi).sum(dim=1)
        return self.global_b + bu + bi + dot

class HybridRec(nn.Module):
    def __init__(self, num_users, num_items, k_user=32, k_item=32, genre_dim=19, hidden=128, dropout=0.1):
        super().__init__()
        self.user_e = nn.Embedding(num_users, k_user)
        self.item_e = nn.Embedding(num_items, k_item)
        self.mlp = nn.Sequential(
            nn.Linear(k_user + k_item + genre_dim + 1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )

    def forward(self, u, i, g, s):
        ue = self.user_e(u)
        ie = self.item_e(i)
        if s.dim() == 1:
            s = s.unsqueeze(1)
        x = torch.cat([ue, ie, g, s], dim=1)
        return self.mlp(x).squeeze(1)

# -------------------------
# Load artifacts + build embeddings (cache-safe)
# -------------------------
ART = Path("artifacts")

@st.cache_resource
def load_all():
    with open(ART / "meta.json", "r") as f:
        meta = json.load(f)

    items_df = pd.read_csv(ART / "items_lookup.csv")
    items_df["i_idx"] = items_df["i_idx"].astype(int)

    num_users = meta["num_users"]
    num_items = meta["num_items"]

    # ---- Load MF and extract item embeddings
    mf_k = meta["mf"]["k"]
    mf = MF(num_users=num_users, num_items=num_items, k=mf_k)
    mf.load_state_dict(torch.load(ART / "mf_deploy.pth", map_location="cpu")["model_state_dict"])
    mf.eval()
    emb_mf = mf.item_f.weight.detach().cpu().numpy()
    del mf  # free memory

    # ---- Load Hybrid and extract item embeddings
    hy_cfg = meta["hybrid"]
    hy = HybridRec(
        num_users=num_users, num_items=num_items,
        k_user=hy_cfg["k_user"], k_item=hy_cfg["k_item"],
        genre_dim=hy_cfg["genre_dim"], hidden=hy_cfg["hidden"], dropout=hy_cfg["dropout"]
    )
    hy.load_state_dict(torch.load(ART / "hybrid_deploy.pth", map_location="cpu")["model_state_dict"])
    hy.eval()
    emb_hy = hy.item_e.weight.detach().cpu().numpy()
    del hy

    # Normalize for cosine similarity
    def _norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    emb_mf_norm = _norm(emb_mf)
    emb_hy_norm = _norm(emb_hy)

    # Best flag from meta (fallback to hybrid)
    best = str(meta.get("best_model", "")).strip().lower()
    if best not in {"mf", "hybrid"}:
        best = "hybrid"

    return meta, items_df, best, emb_mf_norm, emb_hy_norm

meta, items_df, best_model_flag, emb_mf_norm, emb_hy_norm = load_all()

# -------------------------
# Similarity helpers
# -------------------------
def topk_similar_items(i_idx: int, K: int, emb_norm_matrix: np.ndarray):
    v = emb_norm_matrix[i_idx]
    sims = emb_norm_matrix @ v
    sims[i_idx] = -1.0  # exclude self
    if K >= len(sims):
        top_idx = np.argsort(-sims)
    else:
        top_idx = np.argpartition(-sims, K)[:K]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
    scores = sims[top_idx]
    return top_idx.tolist(), scores.tolist()

def title_matches(query: str, df: pd.DataFrame, col="title"):
    q = query.strip().lower()
    if not q:
        return df.iloc[[]]
    cand = df[df[col].str.lower().str.contains(q, na=False)].copy()
    if cand.empty:
        return cand
    # Prefer titles that start with the query
    cand["__rank"] = (~cand[col].str.lower().str.startswith(q)).astype(int)
    cand = cand.sort_values(["__rank", col]).drop(columns="__rank")
    return cand

def pick_embeddings(choice: str):
    """
    choice: 'Auto (best)' | 'MF' | 'Hybrid'
    returns (emb_norm, source_name)
    """
    if choice == "MF":
        return emb_mf_norm, "MF"
    if choice == "Hybrid":
        return emb_hy_norm, "Hybrid"
    # Auto
    return (emb_mf_norm, "MF") if best_model_flag == "mf" else (emb_hy_norm, "Hybrid")

# -------------------------
# UI — Similar by Movie Title only
# -------------------------
st.title("🎬 Similar Movies by Title")

# Sidebar controls
model_choice = st.sidebar.radio("Embeddings to use", ["Auto (best)", "MF", "Hybrid"], index=0)
k_choice = st.sidebar.radio("Top-K", [10, 20], index=0)

emb_norm, src_name = pick_embeddings(model_choice)
auto_note = f"(Auto picked **{best_model_flag.upper()}**)" if model_choice.startswith("Auto") else ""
st.caption(f"Using **{src_name}** item embeddings for cosine similarity. {auto_note}")

# Search + recommend
query = st.text_input("Type a movie title (partial OK)")
if query:
    matches = title_matches(query, items_df, col="title")
    if matches.empty:
        st.info("No titles matched. Try a different query or check spelling.")
    else:
        options = (matches["title"] + "  —  (i_idx=" + matches["i_idx"].astype(str)
                   + ", movie_id=" + matches["movie_id"].astype(str) + ")").tolist()
        pick = st.selectbox("Select a title:", options, index=0)
        sel_row = matches.iloc[options.index(pick)]
        sel_idx = int(sel_row["i_idx"])
        sel_title = sel_row["title"]

        if st.button(f"Get Top-{k_choice} Similar Movies"):
            top_idx, sims = topk_similar_items(sel_idx, int(k_choice), emb_norm)
            out = pd.DataFrame({"i_idx": top_idx, "similarity": sims}).merge(items_df, on="i_idx", how="left")
            st.subheader(f"Top-{k_choice} similar to: “{sel_title}”")
            st.dataframe(out[["i_idx","title","movie_id","similarity"]], use_container_width=True)
