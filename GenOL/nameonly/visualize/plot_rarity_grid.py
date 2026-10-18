# Rarity score line plot용
import matplotlib.pyplot as plt
import pickle

classes = ['dog', 'elephant', 'guitar', 'giraffe', 'horse', 'house']
classes = ['dog', 'horse', 'guitar']
with open('RMD_scores/Rarity_scores_PACS_ablation_plot.pkl', 'rb') as f:
    rarity_scores_dict = pickle.load(f)

p_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# base_dog_scores = [40.77694284555973, 39.66546065310623, 39.20180042035375, 38.46711778114118, 38.25435705949069, 37.882757392783965, 37.64636381925335, 37.1492888424357, 36.91488757814067, 36.47460949673114]
# base_elephant_scores = [37.48073229915178, 37.48073229915178, 37.48073229915178, 37.25177487627783, 37.13729616484085, 37.09913659436185, 37.04363176457423, 37.02281745340387, 36.990109250136165, 36.923593664549486]
# base_guitar_scores = [42.91090216504059, 42.28573349214039, 41.630982492693605, 41.30805000849506, 40.72477112706016, 40.44877820808326, 40.0623881215156, 39.921882635490995, 39.63532655882442, 39.49006812598492]
# base_giraffe_scores = [37.58484389347004, 37.58484389347004, 36.1253687064599, 35.72756294844429, 35.17782915059451, 34.98974585466208, 34.72642924035668, 34.630677744245624, 34.483367750228616, 34.425495966864794]
# base_horse_scores = [41.877711442367314, 41.877711442367314, 41.33432087176928, 40.79093030117124, 40.63567585242894, 40.428669920772535, 40.29693887335483, 40.24753973057319, 40.146054561074564, 40.10546049327512]
# base_house_scores = [28.77186015885015, 28.271706665203272, 27.80429383142341, 27.459665364355942, 26.866758752127904, 26.623515013777936, 26.445136272321296, 26.30872899944269, 26.155270817454255, 26.07621660249051]

# full_dog_scores = [49.24400652498093, 48.796643939149995, 47.534416446019016, 46.064816202031956, 45.644930418035656, 44.977472763924034, 44.73292892389292, 44.36611316384625, 44.157771001347186, 43.824423541348686]
# full_elephant_scores = [38.7198983255682, 38.04650472902046, 37.962330529452, 37.720772275775666, 37.59999314893749, 37.55973343999144, 37.50117386334263, 37.47921402209933, 37.428192851792566, 37.375478231753625]
# full_guitar_scores = [43.581492629165034, 43.581492629165034, 43.581492629165034, 43.581492629165034, 43.581492629165034, 43.581492629165034, 42.868458117672205, 42.609172840765716, 42.13039579006946, 41.926236150302955]
# full_giraffe_scores = [41.627838790045445, 40.337675001679344, 39.32692627753549, 38.9785098007224, 38.58031954150744, 38.11474000142, 37.42315081805663, 37.15358301172405, 36.73886330967394, 36.575937712439966]
# full_horse_scores = [44.60355937168451, 42.609623053680934, 42.262016549651555, 41.799602661614315, 41.66748440788939, 41.258726233895075, 40.880913348046036, 40.71900136377564, 40.46440203596573, 40.344207920406795]
# full_house_scores = [29.18956883222771, 28.59032263893693, 28.097872304604678, 27.85164713743855, 27.2661445467991, 26.948845419428835, 26.519572375951725, 25.861592025115684, 22.929738078911114, 20.84521643537374]

base_scores = {cls: rarity_scores_dict['class'][cls]['base'] for cls in classes}
full_scores = {cls: rarity_scores_dict['class'][cls]['full'] for cls in classes}
base_average_score = rarity_scores_dict['average']['base']
full_average_score = rarity_scores_dict['average']['full']

fig, axs = plt.subplots(2, 2, figsize=(10 * 2, 7 * 2), constrained_layout=True)

for i, cls in enumerate(classes):
    row_idx = i // 2; col_idx = i % 2
    ax = axs[row_idx, col_idx]
    ax.plot(p_values, base_scores[cls], '-o', label='PACS Base Prompt', color='lightsteelblue', linewidth=5, markersize=10)
    ax.plot(p_values, full_scores[cls], '-o', label='PACS Prompt Rewrites', color='indigo', linewidth=5, markersize=10)
    
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    
    # 각 subplot에 레이블, 타이틀 추가
    ax.set_xlabel('p(%)', fontsize=30)
    ax.set_ylabel("Average Score", fontsize=30)
    ax.set_title(f"{cls}", fontsize=30, pad=15)

# Plot average
ax = axs[1, 1]
ax.plot(p_values, base_average_score, '-o', label='PACS Base Prompt', color='lightsteelblue', linewidth=5, markersize=10)
ax.plot(p_values, full_average_score, '-o', label='PACS Prompt Rewrites', color='indigo', linewidth=5, markersize=10)

ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)

# 각 subplot에 레이블, 타이틀 추가
ax.set_xlabel('p(%)', fontsize=30)
ax.set_ylabel("Average Score", fontsize=30)
ax.set_title(f"Average", fontsize=30, pad=15)

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1), fontsize=40)

fig.suptitle('Average Rarity score for top p% samples', fontsize=40, y=1.05)

plt.savefig('test.png')
plt.savefig(f"./vis/average_grid.pdf", bbox_inches="tight")