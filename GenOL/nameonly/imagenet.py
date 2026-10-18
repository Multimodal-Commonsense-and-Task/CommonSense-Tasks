import imagenet_stubs
from collections import defaultdict
from imagenet_stubs.imagenet_2012_labels import label_to_name

label_indices = [i for i in range(1000)]
labels = [label_to_name(label) for label in label_indices]
label_postprocessed = []

count_dict = defaultdict(int)
for i, label in enumerate(labels):
    label_split = label.split(',')
    label_split = [label.strip() for label in label_split]
    count_dict[len(label_split)] += 1
    
    if len(label_split) == 1:
        label_postprocessed.append(label_split[0])
    else:
        # Use two labels for now
        label_postprocessed.append(label_split[0] + ', ' + label_split[1])

result_dict = {}
SAMPLE_COUNT = 300
for i, label in enumerate(label_postprocessed):
    result_dict[str(i)] = 200
breakpoint()