import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
from densify import simulate_densification

GESTURE_CLASSES = ['knock', 'lswipe', 'rswipe', 'rotate']
GESTURE_TO_IDX = {g: i for i, g in enumerate(GESTURE_CLASSES)}

DATA_ROOT = os.environ.get('GESTURE_DATA_ROOT', 'data/')


def load_csv_gestures(csv_path, chunk_frames=30):
    df = pd.read_csv(csv_path, header=None, skiprows=1)
    df.columns = ['frame_id', 'num_obj', 'x', 'y', 'z', 'doppler', 'intensity']
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    gestures = []
    current_gesture = []
    prev_frame = -1

    for _, row in df.iterrows():
        frame_id = int(row['frame_id'])
        if prev_frame != -1 and (frame_id - prev_frame) > 10:
            if len(current_gesture) > 0:
                gestures.append(np.array(current_gesture))
                current_gesture = []
        current_gesture.append([row['x'], row['y'], row['z'], row['doppler']])
        prev_frame = frame_id

    if len(current_gesture) > 0:
        gestures.append(np.array(current_gesture))

    # Continuous recordings (no frame gaps) get chunked into fixed windows
    if len(gestures) == 1 and len(df) > 200:
        gesture_arr = gestures[0]
        frame_ids = df['frame_id'].values.astype(int)
        unique_frames = np.unique(frame_ids)
        gestures = []
        for start in range(0, len(unique_frames), chunk_frames):
            chunk_frame_ids = unique_frames[start:start + chunk_frames]
            mask = np.isin(frame_ids, chunk_frame_ids)
            chunk_points = gesture_arr[mask[:len(gesture_arr)]]
            if len(chunk_points) >= 5:
                gestures.append(chunk_points)

    return gestures


def augment_points(points):
    points = points.copy()
    points[:, :3] += np.random.normal(0, 0.02, points[:, :3].shape)
    scale = np.random.uniform(0.9, 1.1)
    points[:, :3] *= scale
    theta = np.random.uniform(0, 2 * np.pi)
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
    points[:, :3] = points[:, :3] @ R.T
    return points


def pad_or_sample(points, num_points=64):
    n = len(points)
    if n == 0:
        return np.zeros((num_points, 4))
    if n >= num_points:
        idx = np.random.choice(n, num_points, replace=False)
    else:
        idx = np.random.choice(n, num_points, replace=True)
    return points[idx]


class GestureDataset(Dataset):
    def __init__(self, data_root=DATA_ROOT, num_points=64, dense_n=256,
                 train=True, train_split=0.8, augment=True, test_users=None):
        self.num_points = num_points
        self.dense_n = dense_n
        self.augment = augment and train
        self.train = train
        self.samples = []
        self.test_users = test_users

        for user_folder in sorted(os.listdir(data_root)):
            user_path = os.path.join(data_root, user_folder)
            if not os.path.isdir(user_path):
                continue

            if self.test_users is not None:
                if train and user_folder in self.test_users:
                    continue
                if not train and user_folder not in self.test_users:
                    continue

            for gesture in GESTURE_CLASSES:
                csv_file = os.path.join(
                    user_path, f'long_point_{user_folder}_{gesture}.csv')
                if not os.path.exists(csv_file):
                    continue
                gestures = load_csv_gestures(csv_file)
                label = GESTURE_TO_IDX[gesture]
                for g in gestures:
                    if len(g) >= 5:
                        self.samples.append((g, label))

        if self.test_users is None:
            np.random.seed(42)
            idx = np.random.permutation(len(self.samples))
            split = int(len(idx) * train_split)
            if train:
                idx = idx[:split]
            else:
                idx = idx[split:]
            self.samples = [self.samples[i] for i in idx]

        print(f'{"Train" if train else "Test"} dataset: {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sparse_points, label = self.samples[idx]

        if self.augment:
            sparse_points = augment_points(sparse_points)

        if not self.train:
            np.random.seed(idx)

        dense_points = simulate_densification(sparse_points, target_n=self.dense_n)
        dense_points = pad_or_sample(dense_points, self.num_points)

        return torch.FloatTensor(dense_points), label


if __name__ == '__main__':
    ds = GestureDataset(train=True)
    pts, lbl = ds[0]
    print('Sample shape:', pts.shape, 'Label:', GESTURE_CLASSES[lbl])