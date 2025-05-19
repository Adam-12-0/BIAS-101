#!/usr/bin/env python3
"""
simulate_balance.py

Greedy simulations of debiasing UCF-101 via:
  1) adding candidate YouTube videos that satisfy the 2/3 mitigation rule (weight=5),
  2) removing overrepresented UCF-101 samples in balanced batches (weight=1),
  3) mixing add/remove (mode "both") to greedily maximize bias reduction per step.

Reads input CSVs from data/clip/, writes simulation CSVs to data/clip/simulations/,
and writes PNG plots to visualizations/clip/simulations/.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# ——— Logging Configuration ———
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ——— Paths ———
INPUT_DIR    = 'data/clip'
CSV_OUT_DIR  = os.path.join(INPUT_DIR, 'simulations')
PNG_OUT_DIR  = 'visualizations/clip/simulations'

os.makedirs(CSV_OUT_DIR, exist_ok=True)
os.makedirs(PNG_OUT_DIR, exist_ok=True)

# ——— Utility: Load only final predictions ———
def load_final(path, col_name):
    logger.debug(f"Loading {col_name} from {path}")
    df = pd.read_csv(path, usecols=['video_name', 'final'])
    return df.rename(columns={'final': col_name})

# Baseline UCF-101 data
clip_age  = load_final(os.path.join(INPUT_DIR, 'clip_age.csv'),   'age')
clip_gen  = load_final(os.path.join(INPUT_DIR, 'clip_gender.csv'),'gender')
clip_race = load_final(os.path.join(INPUT_DIR, 'clip_race.csv'),  'race')

# YouTube candidate summary data (updated paths)
sum_age   = load_final(os.path.join(INPUT_DIR, 'clip_age2.csv'),   'age')
sum_gen   = load_final(os.path.join(INPUT_DIR, 'clip_gender2.csv'),'gender')
sum_race  = load_final(os.path.join(INPUT_DIR, 'clip_race2.csv'),'race')

logger.info("Merging baseline and summary data")
original = (
    clip_age.merge(clip_gen, on='video_name')
            .merge(clip_race, on='video_name')
)
summary = (
    sum_age.merge(sum_gen, on='video_name')
           .merge(sum_race, on='video_name')
)

# ——— Bias Categories ———
AGE_CATS   = ['kid', 'young', 'old']
GEN_CATS   = ['male', 'female']
RACE_CATS  = ['white', 'asian', 'black', 'middle eastern', 'indian', 'hispanic']

# ——— Chi-square Utilities ———
def chi_square(df, col, categories):
    counts = df[col].value_counts()
    observed = [counts.get(c, 0) for c in categories]
    expected = [sum(observed) / len(categories)] * len(categories)
    score = chi2_contingency([observed, expected])[0]
    logger.debug(f"Chi-square for {col}: {score:.2f}")
    return score


def avg_chi(df_age, df_gen, df_race):
    scores = [
        chi_square(df_age,  'age',    AGE_CATS),
        chi_square(df_gen,  'gender', GEN_CATS),
        chi_square(df_race, 'race',   RACE_CATS)
    ]
    avg = np.mean(scores)
    logger.debug(f"Average chi-square: {avg:.2f}")
    return avg

# ——— Step 1: Evaluate Candidates (2/3 Rule) ———
def evaluate_candidates(base_age, base_gen, base_race, summary_df):
    logger.info("Evaluating candidate videos for 2/3 mitigation rule")
    records = []
    curr_age, curr_gen, curr_race = base_age.copy(), base_gen.copy(), base_race.copy()
    base_scores = {
        'age':    chi_square(curr_age,  'age',    AGE_CATS),
        'gender': chi_square(curr_gen,  'gender', GEN_CATS),
        'race':   chi_square(curr_race, 'race',   RACE_CATS)
    }

    for i, (name, grp) in enumerate(summary_df.groupby('video_name'), 1):
        age_val, gen_val, race_val = grp['age'].iloc[0], grp['gender'].iloc[0], grp['race'].iloc[0]
        if race_val.lower() == 'unknown':
            records.append((name, age_val, gen_val, race_val, False))
            continue

        new_age  = pd.concat([curr_age, pd.DataFrame({'video_name':[name]*5, 'age':[age_val]*5})], ignore_index=True)
        new_gen  = pd.concat([curr_gen, pd.DataFrame({'video_name':[name]*5, 'gender':[gen_val]*5})], ignore_index=True)
        new_race = pd.concat([curr_race, pd.DataFrame({'video_name':[name]*5, 'race':[race_val]*5})], ignore_index=True)

        new_scores = {
            'age':    chi_square(new_age,  'age',    AGE_CATS),
            'gender': chi_square(new_gen,  'gender', GEN_CATS),
            'race':   chi_square(new_race, 'race',   RACE_CATS)
        }
        mitigations = sum(new_scores[k] < base_scores[k] for k in base_scores)
        flag = mitigations >= 2
        records.append((name, age_val, gen_val, race_val, flag))

        if flag:
            curr_age, curr_gen, curr_race = new_age, new_gen, new_race
            base_scores = new_scores

    return pd.DataFrame(records, columns=['video_name','age','gender','race','add_flag'])

# ——— Step 2: Greedy Simulation ———
def greedy_simulation(age_df, gen_df, race_df, cand_df, mode='add', weight=5, batch=50, max_steps=3000):
    logger.info(f"Starting '{mode}' simulation")
    df_age, df_gen, df_race = age_df.copy(), gen_df.copy(), race_df.copy()
    history, added, removed = [], set(), set()
    step = 0

    key = {'add':'num_added', 'remove':'num_removed', 'both':'step'}[mode]

    while True:
        chi_now = avg_chi(df_age, df_gen, df_race)
        step_val = len(added) if mode=='add' else len(removed) if mode=='remove' else step
        history.append((step_val, chi_now))

        progress_path = os.path.join(CSV_OUT_DIR, f"{mode}_curve.csv")
        new_row = pd.DataFrame([{key: step_val, 'avg_chi': chi_now}])
        if not os.path.exists(progress_path):
            new_row.to_csv(progress_path, index=False)
        else:
            new_row.to_csv(progress_path, index=False, header=False, mode='a')

        step += 1
        if len(history) >= max_steps:
            break

        best_delta, best_action = 0, None

        if mode in ('add','both'):
            pool = cand_df[(cand_df['add_flag']) & (~cand_df['video_name'].isin(added))]
            for row in pool.itertuples():
                n_age  = pd.concat([df_age, pd.DataFrame({'video_name':[row.video_name]*weight,'age':[row.age]*weight})], ignore_index=True)
                n_gen  = pd.concat([df_gen, pd.DataFrame({'video_name':[row.video_name]*weight,'gender':[row.gender]*weight})], ignore_index=True)
                n_race = pd.concat([df_race, pd.DataFrame({'video_name':[row.video_name]*weight,'race':[row.race]*weight})], ignore_index=True)
                chi_new = avg_chi(n_age, n_gen, n_race)
                delta   = chi_now - chi_new
                if delta > best_delta:
                    best_delta, best_action = delta, ('add', row.video_name, (n_age,n_gen,n_race))

        if mode in ('remove','both'):
            over = {}
            for df, cats, label in [(df_age, AGE_CATS,'age'), (df_gen, GEN_CATS,'gender'), (df_race, RACE_CATS,'race')]:
                total, exp = len(df), len(cats)
                counts = df[label].value_counts()
                over[label] = counts[counts > (total/exp)].index.tolist()

            df_all = pd.DataFrame({'video_name':df_age['video_name'],'age':df_age['age'],'gender':df_gen['gender'],'race':df_race['race']})
            cands = [vid for vid, grp in df_all.groupby('video_name') if sum(grp.iloc[0][['age','gender','race']].isin(over[label]))>=2]
            for vid in cands[:batch]:
                n_age = df_age[df_age['video_name'] != vid]
                n_gen = df_gen[df_gen['video_name'] != vid]
                n_race= df_race[df_race['video_name'] != vid]
                chi_new = avg_chi(n_age, n_gen, n_race)
                delta   = chi_now - chi_new
                if delta > best_delta:
                    best_delta, best_action = delta, ('remove', vid, (n_age,n_gen,n_race))

        if not best_action:
            break

        action, vid, (df_age,df_gen,df_race) = best_action
        if action == 'add': added.add(vid)
        else:              removed.add(vid)

    return pd.DataFrame(history, columns=[key,'avg_chi']), added, removed

# ——— Main ———
def main():
    cand_df = evaluate_candidates(clip_age, clip_gen, clip_race, summary)
    cand_df.to_csv(os.path.join(CSV_OUT_DIR,'candidate_evaluation.csv'),index=False)

    add_df, _, _ = greedy_simulation(clip_age, clip_gen, clip_race, cand_df, mode='add', weight=5)
    rem_df, _, _ = greedy_simulation(clip_age, clip_gen, clip_race, cand_df, mode='remove', weight=1)
    cmb_df, _, _ = greedy_simulation(clip_age, clip_gen, clip_race, cand_df, mode='both')

    for df, name in [(add_df,'addition'),(rem_df,'removal'),(cmb_df,'combined')]:
        csv_path = os.path.join(CSV_OUT_DIR,f"{name}_curve.csv")
        png_path = os.path.join(PNG_OUT_DIR,f"{name}_curve.png")
        df.to_csv(csv_path,index=False)
        fig, ax = plt.subplots()
        ax.plot(df[df.columns[0]],df['avg_chi'],marker='o')
        ax.set_xlabel(df.columns[0]); ax.set_ylabel('Average Chi-Square')
        ax.set_title(f"{name.capitalize()} Simulation")
        fig.tight_layout(); fig.savefig(png_path); plt.close(fig)

if __name__ == '__main__':
    main()
