#!/usr/bin/env python3
"""
simulate_balance.py

Debias UCF-101 by exploring combinations of addition/removal rules:
  - Addition rules: 2/3 and 3/3 underrepresented mitigation
  - Removal rules: 3/3 and 2/3 dominant-class reduction

Performs all add/remove sequences, outputs curves, candidate lists, a summary bar chart of chi vs dataset size,
and a combined line plot of chi trajectories.

Reads from data/clip/, writes CSVs to data/clip/simulations/ and PNGs to visualizations/clip/simulations/.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Logging config
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
INPUT_DIR   = 'data/clip'
SIM_DIR     = os.path.join(INPUT_DIR, 'simulations')
PLOT_DIR    = 'visualizations/clip/simulations'
for d in (SIM_DIR, PLOT_DIR): os.makedirs(d, exist_ok=True)

# Load final predictions
def load_final(path, col):
    df = pd.read_csv(path, usecols=['video_name','final'])
    return df.rename(columns={'final': col})

# Base & summary
clip_age = load_final(os.path.join(INPUT_DIR,'clip_age.csv'),'age')
clip_gen = load_final(os.path.join(INPUT_DIR,'clip_gender.csv'),'gender')
clip_race= load_final(os.path.join(INPUT_DIR,'clip_race.csv'),'race')
summary  = ( load_final(os.path.join(INPUT_DIR,'clip_age2.csv'),'age')
           .merge(load_final(os.path.join(INPUT_DIR,'clip_gender2.csv'),'gender'),on='video_name')
           .merge(load_final(os.path.join(INPUT_DIR,'clip_race2.csv'),'race'),on='video_name') )
original = clip_age.merge(clip_gen,on='video_name').merge(clip_race,on='video_name')

# Categories
age_cats = ['kid','young','old']
gen_cats = ['male','female']
race_cats= ['white','asian','black','middle eastern','indian','hispanic']

# Chi utilities
def chi_square(df,col,cats):
    counts = df[col].value_counts()
    obs = [counts.get(c,0) for c in cats]
    exp = [sum(obs)/len(cats)]*len(cats)
    score = chi2_contingency([obs,exp])[0]
    return score

def avg_chi(df_age,df_gen,df_race):
    return np.mean([chi_square(df_age,'age',age_cats), chi_square(df_gen,'gender',gen_cats), chi_square(df_race,'race',race_cats)])

# Addition simulation generic
def simulate_add(rule_name, k):
    logger.info(f"[ADD {rule_name}] Starting addition simulation with threshold {k}")
    curr_age, curr_gen, curr_race = clip_age.copy(), clip_gen.copy(), clip_race.copy()
    base = {
        'age':    chi_square(curr_age,'age',age_cats),
        'gender': chi_square(curr_gen,'gender',gen_cats),
        'race':   chi_square(curr_race,'race',race_cats)
    }
    records, accepted = [], []
    for i, (name, grp) in enumerate(summary.groupby('video_name'), 1):
        a,g,r = grp.iloc[0][['age','gender','race']]
        if r.lower()=='unknown':
            records.append((name,a,g,r,False)); continue
        na = pd.concat([curr_age, pd.DataFrame({'video_name':[name]*5,'age':[a]*5})],ignore_index=True)
        ng = pd.concat([curr_gen, pd.DataFrame({'video_name':[name]*5,'gender':[g]*5})],ignore_index=True)
        nr = pd.concat([curr_race,pd.DataFrame({'video_name':[name]*5,'race':[r]*5})],ignore_index=True)
        new_scores = {
            'age':    chi_square(na,'age',age_cats),
            'gender': chi_square(ng,'gender',gen_cats),
            'race':   chi_square(nr,'race',race_cats)
        }
        mit = sum(new_scores[attr] < base[attr] for attr in base)
        flag = mit >= k
        records.append((name,a,g,r,flag))
        if flag:
            logger.debug(f"[ADD {rule_name}] #{i} Accepted {name} (mitigations: {mit})")
            accepted.append(name)
            curr_age, curr_gen, curr_race = na, ng, nr
            base = new_scores
    # write candidates
    pd.DataFrame(records, columns=['video_name','age','gender','race','add_flag']).to_csv(os.path.join(SIM_DIR,f'candidate_add_{rule_name}.csv'),index=False)
    # build and plot curve
    curve=[]; df_age,df_gen,df_race = clip_age.copy(),clip_gen.copy(),clip_race.copy()
    for step, name in enumerate(accepted,1):
        grp = summary[summary.video_name==name]
        a,g,r = grp.iloc[0][['age','gender','race']]
        df_age  = pd.concat([df_age, pd.DataFrame({'video_name':[name]*5,'age':[a]*5})],ignore_index=True)
        df_gen  = pd.concat([df_gen, pd.DataFrame({'video_name':[name]*5,'gender':[g]*5})],ignore_index=True)
        df_race = pd.concat([df_race,pd.DataFrame({'video_name':[name]*5,'race':[r]*5})],ignore_index=True)
        curve.append((step, avg_chi(df_age,df_gen,df_race)))
    add_df = pd.DataFrame(curve, columns=['step','avg_chi'])
    add_df.to_csv(os.path.join(SIM_DIR,f'addition_{rule_name}_curve.csv'),index=False)
    plt.figure(); plt.plot(add_df['step'],add_df['avg_chi'],marker='o', label=f'add_{rule_name}');
    plt.xlabel('step'); plt.ylabel('avg_chi'); plt.title('Addition Simulations');
    plt.legend(); plt.savefig(os.path.join(PLOT_DIR,f'addition_{rule_name}_curve.png')); plt.close()
    return df_age,df_gen,df_race,add_df

# Removal simulation generic
def simulate_remove(rule_name, k, df_age, df_gen, df_race):
    logger.info(f"[REMOVE {rule_name}] Starting removal simulation with threshold {k}")
    orig = set(original.video_name)
    curve=[]; removed=[]; step=0
    while True:
        chi = avg_chi(df_age,df_gen,df_race)
        curve.append((step,chi))
        if chi<1e-2 or step>=len(orig): break
        da = df_age.age.value_counts().idxmax()
        dg = df_gen.gender.value_counts().idxmax()
        dr = df_race.race.value_counts().idxmax()
        df_all = pd.DataFrame({'video_name':df_age.video_name,'age':df_age.age,'gender':df_gen.gender,'race':df_race.race})
        df_all['match'] = df_all[['age','gender','race']].apply(lambda r: (r['age']==da)+(r['gender']==dg)+(r['race']==dr), axis=1)
        cands = df_all[(df_all.video_name.isin(orig)) & (df_all.match>=k)]
        if cands.empty: break
        vid = cands.iloc[0].video_name; removed.append(vid)
        mask=lambda df: df.video_name==vid
        df_age = df_age[~mask(df_age)].reset_index(drop=True)
        df_gen = df_gen[~mask(df_gen)].reset_index(drop=True)
        df_race= df_race[~mask(df_race)].reset_index(drop=True)
        step+=1
    # write candidates
    pd.DataFrame({'video_name':removed}).to_csv(os.path.join(SIM_DIR,f'candidate_removal_{rule_name}.csv'),index=False)
    # write curve
    rem_df = pd.DataFrame(curve, columns=['step','avg_chi'])
    rem_df.to_csv(os.path.join(SIM_DIR,f'removal_{rule_name}_curve.csv'),index=False)
    plt.figure(); plt.plot(rem_df['step'],rem_df['avg_chi'],marker='o', label=f'rem_{rule_name}');
    plt.xlabel('step'); plt.ylabel('avg_chi'); plt.title('Removal Simulations');
    plt.legend(); plt.savefig(os.path.join(PLOT_DIR,f'removal_{rule_name}_curve.png')); plt.close()
    return df_age,df_gen,df_race,rem_df

# Main combos
if __name__=='__main__':
    results = []  # to store (combo, size, final_chi, curve_df)
    # run all combos
    for add_rule, k_add in [('2_3',2),('3_3',3)]:
        ageA,genA,raceA,add_df = simulate_add(add_rule,k_add)
        for rem_rule, k_rem in [('3_3',3),('2_3',2)]:
            ageR,genR,raceR,rem_df = simulate_remove(rem_rule,k_rem,ageA,genA,raceA)
            combo_name = f'{add_rule}+{rem_rule}'
            final_chi = rem_df['avg_chi'].iloc[-1]
            size = len(ageA) + len(ageR) - len(original)
            results.append((combo_name, size, final_chi, add_df, rem_df))
    # Bar chart of all combos
    names = [r[0] for r in results]
    sizes = [r[1] for r in results]
    chis  = [r[2] for r in results]
    plt.figure(figsize=(10,4))
    plt.bar(range(len(results)), chis, tick_label=[f"{n}\n({s})" for n,s in zip(names,sizes)])
    plt.ylabel('Final Avg Chi-Square'); plt.title('All Add/Remove Combos');
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,'all_combos_bar.png')); plt.close()
    # Line graph of chi trajectories
    plt.figure(figsize=(10,6))
    for combo_name, size, _, add_df, rem_df in results:
        df_combo = pd.concat([
            add_df.rename(columns={'step':'x'}),
            rem_df.rename(columns={'step':'x'})
        ], ignore_index=True)
        plt.plot(df_combo['x'], df_combo['avg_chi'],
                 label=f"{combo_name} (size={size})")
    plt.xlabel('Step'); plt.ylabel('Avg Chi-Square'); plt.title('Chi Trajectories for All Combos');
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR,'all_combos_lines.png')); plt.close()
    # Save summary CSV
    summary_df = pd.DataFrame([(n,s,c) for n,s,c,_,_ in results], columns=['combo','dataset_size','final_chi'])
    summary_df.to_csv(os.path.join(SIM_DIR,'all_combos_summary.csv'), index=False)
    logger.info('Completed all simulations and summary outputs.')
