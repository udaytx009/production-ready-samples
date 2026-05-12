"""
eval_student_models.py
-----
Evaluates four student models on ASSISTments splits → Table 3.

No external ML dependencies beyond scikit-learn + sentence-transformers.
BKT is implemented from scratch (no pyBKT needed).

Usage:
    pip install sentence-transformers scikit-learn pandas numpy
    python eval_student_models.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# --- Load ---
print("\n── Loading ASSISTments splits -----")
train = pd.read_csv("data/assistments/train.csv")
val   = pd.read_csv("data/assistments/val.csv")
test  = pd.read_csv("data/assistments/test.csv")
all_train = pd.concat([train, val], ignore_index=True)
SKILL_COL = "skill_name" if "skill_name" in train.columns else "skill_id"
print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
print(f"  Skills: {train[SKILL_COL].nunique()}  Students: {train['user_id'].nunique()}")

# --- Metrics ---
def ece(y_true, y_pred, n_bins=10):
    y_true, y_pred = np.array(y_true), np.clip(np.array(y_pred), 1e-6, 1-1e-6)
    bins = np.linspace(0, 1, n_bins+1)
    val  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_pred >= lo) & (y_pred < hi)
        if m.sum(): val += m.sum() * abs(y_true[m].mean() - y_pred[m].mean())
    return val / len(y_true)

def score(name, yt, yp, yt_c=None, yp_c=None):
    yt = np.array(yt); yp = np.clip(np.array(yp, float), 1e-6, 1-1e-6)
    cold = None
    if yt_c is not None and len(yt_c) > 1 and len(np.unique(yt_c)) > 1:
        cold = roc_auc_score(yt_c, np.clip(np.array(yp_c, float), 1e-6, 1-1e-6))
    return {"model": name,
            "AUC":   roc_auc_score(yt, yp),
            "Brier": brier_score_loss(yt, yp),
            "ECE":   ece(yt, yp),
            "ColdAUC": cold}

# cold-start mask
cnts  = all_train.groupby("user_id").size().to_dict()
cmask = test["user_id"].map(lambda u: cnts.get(u, 0) <= 5).values
print(f"  Cold-start rows (≤5 prior): {cmask.sum():,}")

results = []

# --- 1. Majority baseline ---
print("\n[1/4] Majority class baseline ...")
mp = all_train["correct"].mean()
yp = np.full(len(test), mp)
r  = score("Majority Baseline", test["correct"], yp,
           test["correct"].values[cmask], yp[cmask])
results.append(r)
print(f"  AUC={r['AUC']:.4f}  Brier={r['Brier']:.4f}  ECE={r['ECE']:.4f}")

# --- 2. Standard BKT (self-contained EM) ---
print("\n[2/4] Standard BKT (self-contained EM) ...")

def fit_bkt(responses, n_iter=40):
    pi, pl, pg, ps = 0.3, 0.10, 0.25, 0.10
    for _ in range(n_iter):
        gs = sl = 0.0
        pk = pi
        for r in responses:
            pc = pk*(1-ps) + (1-pk)*pg
            pp = pk*(1-ps)/max(pc,1e-9) if r else pk*ps/max(1-pc,1e-9)
            pp = np.clip(pp, 0, 1)
            sl += pp*(1-r); gs += (1-pp)*r
            pk  = pp + (1-pp)*pl
        n  = max(len(responses)*0.5, 1)
        ps = np.clip(sl/n, 0.01, 0.40)
        pg = np.clip(gs/n, 0.01, 0.45)
    return pi, pl, pg, ps

def bkt_pred_seq(responses, pi, pl, pg, ps):
    out = []; pk = pi
    for r in responses:
        pc = pk*(1-ps) + (1-pk)*pg; out.append(pc)
        pp = pk*(1-ps)/max(pc,1e-9) if r else pk*ps/max(1-pc,1e-9)
        pk = np.clip(pp,0,1) + (1-np.clip(pp,0,1))*pl
    return out

params = {}
for sk, g in all_train.groupby(SKILL_COL):
    params[sk] = fit_bkt(g["correct"].tolist()) if len(g) >= 10 else None

smeans = all_train.groupby(SKILL_COL)["correct"].mean().to_dict()
gm     = all_train["correct"].mean()

bkt_preds = pd.Series(index=test.index, dtype=float)
for (uid, sk), g in test.groupby(["user_id", SKILL_COL]):
    p = params.get(sk)
    if p:
        vals = bkt_pred_seq(g["correct"].tolist(), *p)
    else:
        vals = [smeans.get(sk, gm)] * len(g)
    bkt_preds.loc[g.index] = vals
bkt_preds = bkt_preds.fillna(gm).values

r = score("Standard BKT", test["correct"], bkt_preds,
          test["correct"].values[cmask], bkt_preds[cmask])
results.append(r)
print(f"  AUC={r['AUC']:.4f}  Brier={r['Brier']:.4f}  ECE={r['ECE']:.4f}")

# --- 3. History-feature LR (DKT proxy) ---
print("\n[3/4] History-feature LR (DKT proxy) ...")

def build_feats(df, sdiff):
    rows = []
    for uid, g in df.groupby("user_id"):
        g  = g.reset_index(drop=True)
        rc = 0
        for i, row in g.iterrows():
            att = i + 1
            rows.append({"orig_idx": row.name if "name" in dir(row) else i,
                         "correct": row["correct"],
                         "rolling_acc": rc/att,
                         "attempt": att,
                         "diff": sdiff.get(row[SKILL_COL], 0.5)})
            rc += row["correct"]
    return pd.DataFrame(rows)

sdiff = (1 - all_train.groupby(SKILL_COL)["correct"].mean()).to_dict()
Ftr = build_feats(all_train, sdiff)
Fte = build_feats(test, sdiff)
cols = ["rolling_acc", "attempt", "diff"]
sc  = StandardScaler()
Xtr = sc.fit_transform(Ftr[cols]); Xte = sc.transform(Fte[cols])
lr  = LogisticRegression(max_iter=300, C=1.0)
lr.fit(Xtr, Ftr["correct"])
lr_p = lr.predict_proba(Xte)[:,1]
r = score("History-LR (DKT proxy)", Fte["correct"], lr_p)
results.append(r)
print(f"  AUC={r['AUC']:.4f}  Brier={r['Brier']:.4f}  ECE={r['ECE']:.4f}")

# --- 4. Embedding fusion ---
print("\n[4/4] Embedding fusion (BKT + sentence-transformer) ...")
print("  Downloading ~22MB model on first run ...")
try:
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")

    def make_text(row):  # NO correctness label — predict from context only
        return f"Attempt {int(row['attempt'])} on skill {row.get(SKILL_COL, 'unknown')}. Rolling acc: {row['rolling_acc']:.2f}."

    Ftr2 = build_feats(all_train, sdiff)
    Ftr2[SKILL_COL] = all_train[SKILL_COL].values[:len(Ftr2)] if len(Ftr2)==len(all_train) else "unknown"
    Fte2 = build_feats(test, sdiff)
    Fte2[SKILL_COL] = test[SKILL_COL].values[:len(Fte2)] if len(Fte2)==len(test) else "unknown"

    def encode_batched(df, batch=512):
        texts = [make_text(r) for _, r in df.iterrows()]
        out   = []
        for i in range(0, len(texts), batch):
            out.append(enc.encode(texts[i:i+batch], show_progress_bar=False))
            if i % (batch*10) == 0:
                print(f"    {min(i+batch,len(texts)):,}/{len(texts):,} encoded")
        return np.vstack(out)

    print("  Encoding train ...")
    Etr = encode_batched(Ftr2)
    print("  Encoding test ...")
    Ete = encode_batched(Fte2)

    Xtr2 = np.hstack([sc.transform(Ftr2[cols]), Etr])
    Xte2 = np.hstack([sc.transform(Fte2[cols]), Ete])
    flr  = LogisticRegression(max_iter=300, C=1.0)
    flr.fit(Xtr2, Ftr2["correct"])
    fp   = flr.predict_proba(Xte2)[:,1]

    cold_f = Fte2["attempt"].values <= 5
    r = score("ATS Hybrid (BKT+Embedding)", Fte2["correct"], fp,
              Fte2["correct"].values[cold_f], fp[cold_f])
    results.append(r)
    print(f"  AUC={r['AUC']:.4f}  Brier={r['Brier']:.4f}  ECE={r['ECE']:.4f}")

except Exception as e:
    print(f"  Skipped embedding model: {e}")
    results.append({"model":"ATS Hybrid (BKT+Embedding)",
                    "AUC":None,"Brier":None,"ECE":None,"ColdAUC":None})

# --- Print TABLE 3 ---
print("\n" + "═"*74)
print("  TABLE 3 — Student model evaluation  ")
print("═"*74)
print(f"  {'Model':<36} {'AUC':>6}  {'Brier':>6}  {'ECE':>6}  {'ColdAUC':>8}")
print("  " + "─"*70)
for r in results:
    fmt = lambda v: f"{v:.4f}" if v is not None else "  N/A  "
    print(f"  {r['model']:<36} {fmt(r['AUC']):>6}  {fmt(r['Brier']):>6}"
          f"  {fmt(r['ECE']):>6}  {fmt(r['ColdAUC']):>8}")
print("═"*74)

with open("results/table3.json","w") as f:
    json.dump(results, f, indent=2)
print("\n  Saved → results/table3.json")
print("✓ Done. Results saved to results/ directory.\n")
