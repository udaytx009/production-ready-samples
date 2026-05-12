"""
synthetic_generator.py
-----
Generates two synthetic classroom cohorts for sequencing and
feedback ablation experiments (Section 5.1 of the paper).

Cohort A — Low engagement  (noise=0.3, engagement_dropout=0.25)
Cohort B — High engagement (noise=0.1, engagement_dropout=0.05)

Each cohort: 500 students × 30 skills × ~170 interactions each.

Usage:
    python synthetic_generator.py

Output (all in ./data/synthetic/):
    cohort_low.parquet
    cohort_high.parquet
    stats.json          counts for Table 1
    skill_graph.json    prerequisite structure
"""

import json, os
import numpy as np
import pandas as pd

# --- Reproducibility ---
SEED = 42
rng  = np.random.default_rng(SEED)

# --- Parameters ---
N_STUDENTS   = 500
N_SKILLS     = 30
MAX_STEPS    = 200    # max interactions per student
MASTERY_THR  = 0.85   # student "stops" a skill when true mastery ≥ this

COHORT_CONFIGS = {
    "low":  dict(noise=0.30, engagement_dropout=0.25,
                 prior_alpha=1.5, prior_beta=6.0,   # low prior mastery
                 learn_mean=0.08, learn_std=0.03),
    "high": dict(noise=0.10, engagement_dropout=0.05,
                 prior_alpha=2.5, prior_beta=4.0,   # higher prior mastery
                 learn_mean=0.12, learn_std=0.03),
}

# --- Prerequisite graph (linear chains + some branches) ---
def build_prereq_graph(n_skills):
    """
    Returns dict: skill_id → list of prerequisite skill_ids.
    Structure: 6 chains of 5 skills + 5 "junction" skills.
    """
    graph = {i: [] for i in range(n_skills)}
    # 6 chains of 5
    for chain in range(6):
        base = chain * 5
        for pos in range(1, 5):
            if base + pos < n_skills:
                graph[base + pos] = [base + pos - 1]
    # junction skills depend on end of two chains
    junctions = [25, 26, 27, 28, 29]
    sources    = [(4, 9), (9, 14), (14, 19), (19, 24), (4, 14)]
    for j, (s1, s2) in zip(junctions, sources):
        if j < n_skills and s1 < n_skills and s2 < n_skills:
            graph[j] = [s1, s2]
    return graph

# --- BKT simulation per student ---
def simulate_student(student_id, cfg, prereq_graph, skill_difficulties):
    """
    Simulates a student working through skills.
    Returns a list of interaction records.
    """
    n   = cfg["noise"]
    do  = cfg["engagement_dropout"]

    # Per-student per-skill true mastery (Beta prior)
    mastery = rng.beta(cfg["prior_alpha"], cfg["prior_beta"], size=N_SKILLS)

    # Per-student per-skill learning rate
    learn_rates = np.clip(
        rng.normal(cfg["learn_mean"], cfg["learn_std"], size=N_SKILLS),
        0.01, 0.30
    )

    # Misconception flags — some students have specific misconceptions
    misconceptions = rng.random(N_SKILLS) < 0.15  # 15% skill misconception rate

    records  = []
    step     = 0
    skill_attempts = np.zeros(N_SKILLS, dtype=int)

    while step < MAX_STEPS:
        # Pick an available skill (prereqs met, not yet mastered)
        available = [
            k for k in range(N_SKILLS)
            if mastery[k] < MASTERY_THR
            and all(mastery[p] >= MASTERY_THR for p in prereq_graph.get(k, []))
        ]
        if not available:
            break  # all skills mastered

        # Engagement dropout — student skips this step
        if rng.random() < do:
            step += 1
            continue

        skill = rng.choice(available)
        skill_attempts[skill] += 1

        # Observed correctness with slip/guess noise
        p_correct = mastery[skill] * (1 - n) + (1 - mastery[skill]) * (n * 0.3)
        if misconceptions[skill]:
            p_correct *= 0.6   # misconception suppresses performance

        correct = int(rng.random() < p_correct)

        # Engagement features
        time_on_task   = max(10, rng.normal(90, 40))   # seconds
        hints_requested = int(not correct and rng.random() < 0.4)
        attempt_num    = skill_attempts[skill]

        records.append({
            "user_id":         student_id,
            "skill_id":        skill,
            "step":            step,
            "correct":         correct,
            "true_mastery":    round(float(mastery[skill]), 4),
            "difficulty":      round(float(skill_difficulties[skill]), 4),
            "time_on_task_s":  round(float(time_on_task), 1),
            "hints_requested": hints_requested,
            "attempt_num":     attempt_num,
            "has_misconception": int(misconceptions[skill]),
        })

        # Update true mastery after practice
        gain = learn_rates[skill] * (1 - mastery[skill])
        if correct:
            mastery[skill] = min(1.0, mastery[skill] + gain)
        else:
            mastery[skill] = max(0.0, mastery[skill] - gain * 0.2)

        step += 1

    return records

def generate_cohort(name, cfg, prereq_graph, skill_difficulties):
    print(f"\n  Generating cohort '{name}' ({N_STUDENTS} students) ...")
    all_records = []
    for sid in range(N_STUDENTS):
        all_records.extend(simulate_student(sid, cfg, prereq_graph, skill_difficulties))
        if (sid + 1) % 100 == 0:
            print(f"    {sid+1}/{N_STUDENTS} students done")

    df = pd.DataFrame(all_records)
    print(f"  Total interactions: {len(df):,}")
    print(f"  Avg per student:    {len(df)/N_STUDENTS:.0f}")
    return df

def print_and_save_stats(dfs, out_dir):
    stats = {}
    print("\n" + "═"*56)
    print("  TABLE 1 — Synthetic cohort stats  (copy into paper)")
    print("═"*56)
    for name, df in dfs.items():
        s = {
            "students":            int(df["user_id"].nunique()),
            "interactions":        int(len(df)),
            "skills":              int(df["skill_id"].nunique()),
            "avg_per_student":     round(len(df)/df["user_id"].nunique(), 1),
            "overall_accuracy":    round(float(df["correct"].mean()), 4),
        }
        stats[f"synthetic_{name}"] = s
        print(f"\n  Cohort: {name.upper()}")
        for k, v in s.items():
            print(f"    {k:<30} {v}")
    print("═"*56)
    with open(f"{out_dir}/stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats

def main():
    out_dir = "data/synthetic"
    os.makedirs(out_dir, exist_ok=True)

    print("\n── Synthetic Classroom Generator -----")
    print(f"   Seed={SEED}  Students={N_STUDENTS}  Skills={N_SKILLS}")

    prereq_graph      = build_prereq_graph(N_SKILLS)
    skill_difficulties = rng.beta(2, 5, size=N_SKILLS)  # most skills moderate difficulty

    # Save prerequisite graph
    with open(f"{out_dir}/skill_graph.json", "w") as f:
        json.dump({str(k): v for k, v in prereq_graph.items()}, f, indent=2)
    print(f"\n  Prerequisite graph saved → {out_dir}/skill_graph.json")

    dfs = {}
    for name, cfg in COHORT_CONFIGS.items():
        df = generate_cohort(name, cfg, prereq_graph, skill_difficulties)
        path = f"{out_dir}/cohort_{name}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved → {path}")
        dfs[name] = df

    print_and_save_stats(dfs, out_dir)
    print("\n✓ Done.

if __name__ == "__main__":
    main()
