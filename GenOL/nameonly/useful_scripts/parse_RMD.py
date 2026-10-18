import os
import pickle
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pickle_path = './RMD_scores_domainnet.pkl'
figure_path = os.path.basename(pickle_path).replace('.pkl', '.png')

topk = 3
RMD_scores = pickle.load(open(pickle_path, 'rb')) # (model, cls) -> [(image_path, RMD_score), ...]

# Calculate the average RMD score of each model, class pair.
classes = []
for (model, cls) in RMD_scores.keys():
    if cls not in classes:
        classes.append(cls)

RMD_scores_per_class = {cls: {} for cls in classes}

# Calculate the average RMD score for each class, model
for (model, cls), images in RMD_scores.items():
    score_list = [score for _, score in images]
    try:
        avg_score = sum(score_list) / len(score_list)
    except ZeroDivisionError:
        avg_score = 0
    RMD_scores_per_class[cls][model] = avg_score

    # Print the top-k, bottom-k images
    print(f"Top {topk} images for class {cls}, model {model}:")
    topk_images = sorted(images, key=lambda x: x[1], reverse=True)[:topk]
    bottomk_images = sorted(images, key=lambda x: x[1])[:topk]

    print(f"Top {topk} images of class {cls}, model {model}:")
    for image, score in topk_images:
        print(f"{image}, RMD score: {score}")

    print(f"Bottom {topk} images of class {cls}, model {model}:")
    for image, score in bottomk_images:
        print(f"{image}, RMD score: {score}")

# Temp - check if cogview2 has the largest RMD score
for cls in RMD_scores_per_class.keys():
    cls_model_scores = RMD_scores_per_class[cls] # {model: score, ...}
    # Find the model with the largest RMD score
    # max_model = max(cls_model_scores, key=cls_model_scores.get)
    # min_model = min(cls_model_scores, key=cls_model_scores.get)
    # Check the range of RMD scores
    sdxl_score = cls_model_scores['sdxl']; dalle2_score = cls_model_scores['dalle2']
    floyd_score = cls_model_scores['floyd']; cogview2_score = cls_model_scores['cogview2']

    if 20 < sdxl_score < 50 and 0 < floyd_score < 50 and -50 < dalle2_score < 0:
        print(f"Class {cls}, sdxl: {sdxl_score}, floyd: {floyd_score}, dalle2: {dalle2_score}, cogview2: {cogview2_score}")
        breakpoint()


# Print the average RMD score for each class, model
for cls in RMD_scores_per_class.keys():
    print(f"Average RMD score for class {cls}:")
    for model in RMD_scores_per_class[cls].keys():
        print(f"Model {model}: {RMD_scores_per_class[cls][model]}")

# Visualization
num_classes = 5
# Randomly choose the classes from the dictionary and slice it
classes = random.choices(list(RMD_scores_per_class.keys()), k=num_classes)
RMD_scores_per_class = {cls: RMD_scores_per_class[cls] for cls in classes}

for cls in classes:
    print(f"Average RMD score for class {cls}:")
    for model in RMD_scores_per_class[cls].keys():
        print(f"Model {model}: {RMD_scores_per_class[cls][model]}")

# lasses = ['marker', 'camouflage', 'stairs', 'blackberry', 'candle', 'anvil']
RMD_scores_per_class = {cls: RMD_scores_per_class[cls] for cls in classes}
df = pd.DataFrame(RMD_scores_per_class).T.reset_index()
df = df.rename(columns={'index': 'class'})
df_melt = df.melt(id_vars=["class"], var_name="model", value_name="score")

plt.figure(figsize=(12, 8))
sns.barplot(x='score', y='class', hue='model', data=df_melt)
plt.title(f'Model Score by Class, {pickle_path}')
plt.xlabel('Score')
plt.ylabel('Class')
plt.legend(title='Model', loc='lower right')
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# Save the figure
plt.savefig(figure_path, dpi=300)

breakpoint()