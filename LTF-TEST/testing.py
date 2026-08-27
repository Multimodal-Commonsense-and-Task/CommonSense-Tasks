import json
from collections import defaultdict
import scienceplots
import matplotlib.pyplot as plt

plt.style.use(['science','ieee'])
def extract_scores(json_line):
    eval_raw = json_line['eval_raw']
    # Intro & Title, Reasons, Conclusions, RtA

    for idx, line in enumerate(eval_raw.split('\n')):
        if 'Title & Intro' in line:
            title_intro = int(line.split(': ')[1])
        elif 'Reasons' in line:
            reasons = int(line.split(': ')[1])
        elif 'Conclusion' in line:
            conclusions = int(line.split(': ')[1])
        elif 'RtA' in line:
            rta = int(line.split(': ')[1])
    mean = (title_intro + reasons + conclusions) / 3

    # return int(mean), rta
    # mean을 반올림
    return round(mean), rta

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data

lines = read_jsonl('mistral_None_ReFair_Cross_EVAL.jsonl')
results = defaultdict(list)
for idx, line in enumerate(lines):
    json_line = json.loads(line)
    try: 
        score, rta = extract_scores(json_line)
        results[json_line['group1'] + "_" + json_line['group2']].append(score)
    except:
        pass

# mean for all
for key, value in results.items():
    results[key] = round(sum(value) / len(value), 2)

print(results)

# draw energy map
import seaborn as sns
import pandas as pd
import numpy as np
sequences = ["white men", "white women", "black men", "black women", "asian men", "asian women", "hispanic men", "hispanic women"]
sequences_short = ["WM", "WW", "BM", "BW", "AM", "AW", "HM", "HW"]
N = len(sequences)
energy_map = np.zeros((N, N))

fig, ax = plt.subplots()

for key, value in results.items():
    x, y = key.split('_')
    x = sequences.index(x)
    y = sequences.index(y)
    energy_map[x][y] = value

ax.imshow(energy_map, cmap='Greens', vmin=-1, vmax=2)
# plt.colorbar()
ax.set_xticks(np.arange(len(sequences)))
ax.set_yticks(np.arange(len(sequences)))
ax.set_xticklabels(sequences_short)
ax.set_yticklabels(sequences_short)
ax.xaxis.tick_top()
ax.xaxis.set_ticks_position('none')
# x tick labels to short
for i in range(N):
    for j in range(N):
        text = plt.text(j, i, energy_map[i, j], ha="center", va="center", color="w", fontsize=8, fontweight='800')

plt.savefig('energy_map.png')
# n x n으로 자동 계산

df = pd.DataFrame(results, index=[0])

# y축이 _ 기준 왼쪽, x축이 _ 기준 오른쪽


plt.figure(figsize=(10, 10))
# 일단 group 별로
