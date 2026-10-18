# Rarity score line plot용
# 지금은 base, full 비교에서 민혁님이 base, equalweight, rmd 이렇게 3개를 비교해달라고 하셔서, 주석처리를 해둔 부분이 있음.
import matplotlib.pyplot as plt
import pickle

classes = ['dog', 'elephant', 'guitar', 'giraffe', 'horse', 'house']
# classes = ['dog', 'giraffe', 'house']
classes = ['horse', 'guitar', 'elephant']
with open('RMD_scores/Rarity_scores_PACS_ablation_plot_new.pkl', 'rb') as f:
    rarity_scores_dict = pickle.load(f)

p_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Base prompt
base_scores = {cls: rarity_scores_dict['class'][cls]['base'] for cls in classes}
base_average_score = rarity_scores_dict['average']['base']

# Prompt rewritten
full_scores = {cls: rarity_scores_dict['class'][cls]['full'] for cls in classes}
full_average_score = rarity_scores_dict['average']['full']

# Equalweight ensemble
equalweight_scores = {cls: rarity_scores_dict['class'][cls]['equalweight'] for cls in classes}
equalweight_average_score = rarity_scores_dict['average']['equalweight']

# RMD ensemble
rmd_scores = {cls: rarity_scores_dict['class'][cls]['rmd'] for cls in classes}
rmd_average_score = rarity_scores_dict['average']['rmd']

# Class average (선택한 class에 대한 average만 하는 것) - 여기를 주석처리 안하면 선택한 class만 average됨.
length = len(base_average_score)
base_average_score = []; equalweight_average_score = []; rmd_average_score = []
for index in range(length):
    base_scores_all_classes = [base_scores[cls][index] for cls in classes]
    equalweight_scores_all_classes = [equalweight_scores[cls][index] for cls in classes]
    rmd_scores_all_classes = [rmd_scores[cls][index] for cls in classes]
    base_average_score.append(sum(base_scores_all_classes) / len(base_scores_all_classes))
    equalweight_average_score.append(sum(equalweight_scores_all_classes) / len(equalweight_scores_all_classes))
    rmd_average_score.append(sum(rmd_scores_all_classes) / len(rmd_scores_all_classes))

fig, axs = plt.subplots(1, len(classes), figsize=(10 * (len(classes)), 7.5), constrained_layout=True)

for i, cls in enumerate(classes):
    ax = axs[i]
    ax.plot(p_values, base_scores[cls], '-o', label='PACS Base Prompt', color='#EBAE44', linewidth=10, markersize=20)
    # ax.plot(p_values, full_scores[cls], '-o', label='PACS Prompt Rewrites', color='indigo', linewidth=10, markersize=20)
    ax.plot(p_values, equalweight_scores[cls], '-o', label='PACS Equalweight Ensemble', color='#A22932', linewidth=10, markersize=20)
    ax.plot(p_values, full_scores[cls], '-o', label='DISCOBER', color='#265195', linewidth=10, markersize=20)
    
    ax.xaxis.set_tick_params(labelsize=40)
    ax.yaxis.set_tick_params(labelsize=40)
    
    # 각 subplot에 레이블, 타이틀 추가
    ax.set_xlabel('p(%)', fontsize=45)
    ax.set_ylabel("Average Score", fontsize=45)
    ax.set_title(f"{cls}", fontsize=60, pad=20)
    # ax.set_ylim((35, 55))

# # Plot average
# ax = axs[-1]
# ax.plot(p_values, base_average_score, '-o', label='PACS Base Prompt', color='lightsteelblue', linewidth=10, markersize=20)
# # ax.plot(p_values, full_average_score, '-o', label='PACS Prompt Rewrites', color='indigo', linewidth=10, markersize=20)
# ax.plot(p_values, equalweight_average_score, '-o', label='PACS Equalweight Ensemble', color='#FFE699', linewidth=10, markersize=20)
# ax.plot(p_values, rmd_average_score, '-o', label='PACS RMD Ensemble', color='#C5E0B4', linewidth=10, markersize=20)

# ax.xaxis.set_tick_params(labelsize=40)
# ax.yaxis.set_tick_params(labelsize=40)

# # 각 subplot에 레이블, 타이틀 추가
# ax.set_xlabel('p(%)', fontsize=45)
# ax.set_ylabel("Average Score", fontsize=45)
# ax.set_title(f"Average", fontsize=60, pad=15)

handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.25), fontsize=45)

fig.suptitle('Average Rarity score for top p% samples', fontsize=65, y=1.14)

plt.savefig('test.png')
plt.savefig(f"./vis/Rarity_Score.pdf", bbox_inches="tight")