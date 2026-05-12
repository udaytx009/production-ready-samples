"""
eval_sequencing.py
-----
Evaluates three sequencing policies on the synthetic cohorts → Table 4.

Policies:
  1. Random          — random unit selection (lower bound)
  2. Greedy          — pick unit with max expected immediate mastery gain
  3. ATS Bandit      — Thompson Sampling contextual bandit

Metrics:
  - Time-to-mastery  (avg units to reach threshold across all students/skills)
  - Normalised Learning Gain  (post−pre)/(1−pre) averaged across students
  - Mastery gain variance     (fairness metric)

Usage:
    python eval_sequencing.py

Output: prints TABLE 4 .
"""

import os, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

MASTERY_THR = 0.80   # threshold to consider a skill "mastered"
MAX_STEPS   = 150    # budget per student
SEED        = 42
rng         = np.random.default_rng(SEED)
N_RUNS      = 5      # repeat each policy N_RUNS times, report mean ± std

print("\n── Loading synthetic cohorts -----")
low  = pd.read_csv("data/synthetic/cohort_low.csv")
high = pd.read_csv("data/synthetic/cohort_high.csv")
skill_graph = json.load(open("data/synthetic/skill_graph.json"))
prereqs = {int(k): v for k, v in skill_graph.items()}

N_SKILLS = int(max(low["skill_id"].max(), high["skill_id"].max())) + 1
print(f"  Low  cohort: {low['user_id'].nunique()} students, {len(low):,} interactions")
print(f"  High cohort: {high['user_id'].nunique()} students, {len(high):,} interactions")
print(f"  Skills: {N_SKILLS}  |  Mastery threshold: {MASTERY_THR}")

# --- Estimate per-skill difficulty from cohort data ---
def skill_difficulty(df):
    return 1 - df.groupby("skill_id")["correct"].mean()

# --- Simulate one student under a given policy ---
def simulate(student_rows, policy_fn, skill_diff, n_steps=MAX_STEPS):
    """
    student_rows: DataFrame of this student's pre-existing interactions
    Returns: (mastery_gains per skill, steps_to_mastery per skill)
    """
    # initialise mastery from cohort data (use true_mastery column)
    init_mastery = student_rows.groupby("skill_id")["true_mastery"].last().to_dict()
    mastery = {s: init_mastery.get(s, 0.1) for s in range(N_SKILLS)}

    # learning rates estimated from data
    learn_rate = {s: 0.08 + 0.04 * (1 - skill_diff.get(s, 0.5))
                  for s in range(N_SKILLS)}

    steps_to_mastery = {}
    step = 0

    while step < n_steps:
        # available skills: prereqs met, not yet mastered
        available = [
            s for s in range(N_SKILLS)
            if mastery[s] < MASTERY_THR
            and all(mastery.get(p, 0) >= MASTERY_THR for p in prereqs.get(s, []))
        ]
        if not available:
            break

        # policy selects unit
        chosen = policy_fn(available, mastery, skill_diff)

        # simulate response
        p_correct = mastery[chosen] * 0.9 + (1 - mastery[chosen]) * 0.2
        correct   = int(rng.random() < p_correct)
        gain      = learn_rate[chosen] * (1 - mastery[chosen])
        mastery[chosen] = min(1.0, mastery[chosen] + (gain if correct else -gain*0.1))

        if mastery[chosen] >= MASTERY_THR and chosen not in steps_to_mastery:
            steps_to_mastery[chosen] = step + 1

        step += 1

    return mastery, steps_to_mastery, init_mastery

# --- Policy definitions ---

def policy_random(available, mastery, skill_diff):
    return rng.choice(available)

def policy_greedy(available, mastery, skill_diff):
    # pick skill with highest expected immediate mastery gain
    gains = []
    for s in available:
        lr   = 0.08 + 0.04 * (1 - skill_diff.get(s, 0.5))
        gain = lr * (1 - mastery[s])
        gains.append(gain)
    return available[int(np.argmax(gains))]

class ThompsonBandit:
    """Simple Bayesian linear bandit with Thompson Sampling."""
    def __init__(self, n_skills):
        self.alpha = np.ones(n_skills)   # Beta distribution alpha (successes)
        self.beta  = np.ones(n_skills)   # Beta distribution beta  (failures)

    def select(self, available, mastery, skill_diff):
        samples = []
        for s in available:
            # sample expected reward = mastery gain / normalised difficulty
            reward_sample = rng.beta(self.alpha[s], self.beta[s])
            # scale by expected gain per unit time
            lr   = 0.08 + 0.04 * (1-skill_diff.get(s, 0.5))
            gain = lr * (1 - mastery[s])
            samples.append(reward_sample * gain)
        return available[int(np.argmax(samples))]

    def update(self, skill, correct):
        if correct:
            self.alpha[skill] += 1
        else:
            self.beta[skill]  += 1

def policy_bandit_factory(n_skills):
    bandit = ThompsonBandit(n_skills)
    def fn(available, mastery, skill_diff):
        return bandit.select(available, mastery, skill_diff)
    return fn, bandit

# --- Run evaluation ---
def eval_policy(name, policy_fn_factory, df, skill_diff, n_runs=N_RUNS):
    all_ttm = []      # time-to-mastery across students
    all_nlg = []      # normalised learning gain
    all_gain = []     # raw mastery gain per student (for variance)

    students = df["user_id"].unique()

    for run in range(n_runs):
        run_ttm = []; run_nlg = []; run_gain = []

        for uid in students:
            srows = df[df["user_id"] == uid]
            if callable(policy_fn_factory):
                try:
                    pf, _ = policy_fn_factory(N_SKILLS)
                    policy_fn = pf
                except:
                    policy_fn = policy_fn_factory
            else:
                policy_fn = policy_fn_factory

            final_mastery, steps_to_mastery, init_mastery = simulate(
                srows, policy_fn, skill_diff)

            # TTM: mean steps across mastered skills
            if steps_to_mastery:
                run_ttm.append(np.mean(list(steps_to_mastery.values())))

            # NLG per student: mean across skills
            nlgs = []
            for s in range(N_SKILLS):
                pre  = init_mastery.get(s, 0.1)
                post = final_mastery.get(s, pre)
                if pre < 1.0:
                    nlgs.append((post - pre) / (1 - pre))
            if nlgs:
                run_nlg.append(np.mean(nlgs))
                run_gain.append(np.mean([final_mastery.get(s,0) - init_mastery.get(s,0.1)
                                         for s in range(N_SKILLS)]))

        all_ttm.append(np.mean(run_ttm)  if run_ttm  else 0)
        all_nlg.append(np.mean(run_nlg)  if run_nlg  else 0)
        all_gain.append(np.var(run_gain) if run_gain else 0)

    return {
        "policy":      name,
        "TTM_mean":    np.mean(all_ttm),
        "TTM_std":     np.std(all_ttm),
        "NLG_mean":    np.mean(all_nlg),
        "NLG_std":     np.std(all_nlg),
        "GainVar":     np.mean(all_gain),
    }

policies = [
    ("Random",           policy_random),
    ("Greedy",           policy_greedy),
    ("ATS Bandit (TS)",  policy_bandit_factory),
]

all_results = []
for cohort_name, cohort_df in [("Low-engagement", low), ("High-engagement", high)]:
    print(f"\n  ── Cohort: {cohort_name} ──")
    sdiff = skill_difficulty(cohort_df).to_dict()
    for pol_name, pol_fn in policies:
        print(f"    Running {pol_name} ({N_RUNS} runs) ...", end=" ", flush=True)
        r = eval_policy(pol_name, pol_fn, cohort_df, sdiff)
        r["cohort"] = cohort_name
        all_results.append(r)
        print(f"TTM={r['TTM_mean']:.1f}  NLG={r['NLG_mean']:.4f}  Var={r['GainVar']:.5f}")

# --- Print TABLE 4 ---
print("\n" + "═"*78)
print("  TABLE 4 — Sequencing policy comparison  ")
print("═"*78)
print(f"  {'Cohort':<17} {'Policy':<22} {'TTM ↓':>7}  {'NLG ↑':>7}  {'GainVar ↓':>10}")
print("  " + "─"*74)
for r in all_results:
    print(f"  {r['cohort']:<17} {r['policy']:<22} "
          f"{r['TTM_mean']:>6.1f}  {r['NLG_mean']:>7.4f}  {r['GainVar']:>10.5f}")
print("═"*78)

with open("results/table4.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\n  Saved → results/table4.json")
print("✓ Done. Results saved to results/ directory.\n")
