import numpy as np


def class_ensemble(id_to_labels, labels_to_onehot, aim_class):
    label_aim = labels_to_onehot[aim_class]
    index = np.argwhere(label_aim == 1)

    aim_id = []
    for key, value in id_to_labels.items():
        if value[index] == 1:
            aim_id.append(key)
    return aim_id


