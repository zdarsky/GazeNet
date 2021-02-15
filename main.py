import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import net
import time
import pandas as pd
import csv

config = {
    "bs": 4,
    "epochs": 15000,
    "momentum": 0.6,
    "lr": 3e-2
}

CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


def get_run_id() -> str:
    n = int(time.time())
    s = ""
    while n > 0:
        s += CHARS[n % len(CHARS)]
        n //= len(CHARS)
    return s[::-1]  # reverse


def decode(s: str) -> int:
    pow = 0
    res = 0
    for c in reversed(s):
        res += s.index(c) * len(s) ** pow
        pow += 1
    return res


def save(path: str, arr: np.ndarray, labels: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter=",")
        w.writerow(labels)
        for row in arr:
            w.writerow(row)


def clean(pose, gaze=None, thres=0.7):
    likes = pose.iloc[:, 2::3]
    idx_uncert = likes[(likes < thres).any(1)].index
    print(f"dropping {len(idx_uncert)} "
          f"({len(idx_uncert)/len(likes)*100:.1f}%) "
          "uncertain frames...")
    pose = pose.drop(idx_uncert)
    # also drop likelihood columns
    pose = pose.drop(likes, axis=1)
    if gaze is not None:
        gaze = gaze.drop(idx_uncert)
    return pose, gaze


def avg_pupil(pose):
    pose = pose.to_numpy()
    print(pose.shape)
    compressed = []
    for row in pose:
        avg_l_x = np.average(row[4:11:2])
        avg_l_y = np.average(row[5:12:2])
        avg_r_x = np.average(row[16:23:2])
        avg_r_y = np.average(row[17:24:2])
        # compressed.append(np.append([row[0:5], avg_x, avg_y], row[10:]))
        # print("building array...")
        a = row[:4]
        a = np.append(a, avg_l_x)
        a = np.append(a, avg_l_y)
        a = np.append(a, row[12:16])
        a = np.append(a, avg_r_x)
        a = np.append(a, avg_r_y)
        a = np.append(a, row[24:])
        compressed.append(a)
        # compressed.append(np.array(
            # [row[:4], avg_l_x, avg_l_y, row[13:17],
             # avg_r_x, avg_r_y, row[25:]]).flatten())
    print(np.array(compressed).shape)
    return pd.DataFrame(compressed)


def shuffle(pose, gaze):
    rnd = np.random.permutation(pose.index)
    return pose.reindex(rnd), gaze.reindex(rnd)


# use as filename to reference more easily
timestamp_enc = get_run_id()
timestamp_enc += "_lu_ld_rrd_0.7"

# FN_TRAIN = ["lu", "ld", "rrd"]
root = "/Users/nzdarsky/data/gazenet/posecollection"
# FIXME: use x_50000.csv for most recent version?
# TODO: test only using lu, ld; improve rrd performance and repeat on all data
# ending = "DLC_resnet50_attentiontestFeb19shuffle1_95000.csv"
ending = "DLC_resnet50_attentiontestFeb19shuffle1_45000.csv"
# ending = "DLC_resnet50_70frames7labelsJun28shuffle1_10000.csv"
FN_POSE = [
    f"{root}/pose_lu{ending}",
    f"{root}/pose_ld{ending}",
    f"{root}/pose_rrd{ending}"
]
FN_GAZE = [
    f"{root}/gazedots_lu.csv",
    f"{root}/gazedots_ld.csv",
    f"{root}/gazedots_rrd.csv"
]


pose = pd.DataFrame()
gaze = pd.DataFrame()
for fn_p, fn_g in zip(FN_POSE, FN_GAZE):
    tmp = pd.read_csv(fn_p, sep=",", skiprows=[0,2], dtype="float")
    # get rid of double index column
    tmp = tmp.drop("bodyparts", axis=1)
    pose = pd.concat([pose, tmp], ignore_index=True)

    tmp = pd.read_csv(fn_g, sep=",")
    gaze = pd.concat([gaze, tmp], ignore_index=True)

print(pose.shape, gaze.shape)

pose, gaze = clean(pose, gaze)
# only drop likelihood columns
# likes = pose.iloc[:, 2::3]
# pose = pose.drop(likes, axis=1)

pose, gaze = shuffle(pose, gaze)
pose = avg_pupil(pose)
pose = pose / 600

pose = pose.to_numpy()
gaze = gaze.to_numpy()

DS_LEN = len(pose)
split_val = int(DS_LEN*1/2)
split_test = int(DS_LEN*3/4)
print(pose.shape, gaze.shape)

# exit("test")
# rng = np.random.RandomState()
# DS_X = train_pose.iloc[:, ]
# DS_Y = train_trial.iloc[:, ]


# idx = np.arange(X.shape[0])
# idx_shuffled = np.random.permutation(idx)

# splits = np.array_split(idx_shuffled, n_folds)

# for i in range(n_folds):
    # splits_cp = copy.deepcopy(splits)  # prolly no deepcopy needed
    # split_test = splits_cp.pop(i)
    # split_train = np.hstack(splits_cp)

    # X_test = X[split_test]
    # y_test = y[split_test]

    # X_train = X[split_train]
    # y_train = y[split_train]
#  DS[0] -> train, DS[1] -> test
DS_X = np.split(pose, [split_val, split_test])
DS_Y = np.split(gaze, [split_val, split_test])


print(f"starting: {timestamp_enc}")
print(f"size of dataset: train: {len(DS_X[0])}, val: {len(DS_X[1])}, "
      f"test: {len(DS_X[2])};")

TRAIN_X = torch.FloatTensor(DS_X[0])
TRAIN_Y = torch.FloatTensor(DS_Y[0])

DS_TRAIN = TensorDataset(TRAIN_X, TRAIN_Y)
# shuffle probably not needed (already shuffled)
DL_TRAIN = DataLoader(DS_TRAIN, config["bs"], shuffle=True)

VALID_X = torch.FloatTensor(DS_X[1])
VALID_Y = torch.FloatTensor(DS_Y[1])

DS_VALID = TensorDataset(VALID_X, VALID_Y)
TEST_X = torch.FloatTensor(DS_X[2])
TEST_Y = torch.FloatTensor(DS_Y[2])

DS_TEST = TensorDataset(TEST_X, TEST_Y)

nw = net.NetWrapper(timestamp_enc, testing=False)
nw.init(config["epochs"], config["lr"], config["momentum"])
nw.set_dataloader(DL_TRAIN)
nw.set_scheduler_inteval(2000)

nw.set_ds(TRAIN_X, TRAIN_Y, VALID_X, VALID_Y)
nw.fit()

labels = ["real_x", "real_y", "pred_x", "pred_y"]
nw.test(TEST_X, TEST_Y)
preds = nw.eval(TEST_X)
preds_with_real = np.hstack((TEST_Y.detach().numpy(), preds.detach().numpy()))
root = "/Users/nzdarsky/code/thesis_bachelor/data"
save(f"{root}/preds/{timestamp_enc}.csv", preds_with_real, labels)
nw.save_best_model()
