import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pyedflib
from tqdm import tqdm
import random
import pandas as pd

train_csv = [[1, 2, 7, 8, 9, 10, 11, 12, 13, 14],
             [1, 2, 3, 4, 5, 6, 11, 12, 13, 14],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
test_csv = [[3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14]]

train_person_csv = [[2, 7, 8, 9, 10, 11, 12, 13, 14],
             [1, 3, 4, 5, 6, 11, 12, 13, 14],
             [2, 3, 4, 5, 6, 7, 8, 9, 10]]
test_person_csv = [[1, 3, 4, 5, 6], [2, 7, 8, 9, 10], [1, 11, 12, 13, 14]]

# EEG Dataset class
class EEGDataset(Dataset):
    def __init__(self, mode, n_channels, channel_indices, trial):
        self.n_channels = n_channels  # 新增参数
        self.channel_indices = channel_indices
        self.trial = trial
        self.data, self.labels, self.filenames = self.load_data(mode, trial)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].transpose((1, 0))  # 确保数据形状为 (channels, sequence_length)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32), self.filenames[idx]

    def load_data(self, mode, trial):
        if trial == 3:
            trial = 0
        # path = r"C:\Projects\person identification\eeg-motor-movementimagery-dataset-1.0.0\files"
        path = r"C:\Projects\person identification\mydata"
        file_list = []
        if mode == "train":
            for i in train_csv[trial]:
                file_list += [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith(f'R{i:02d}.edf')]
        elif mode == "test":
            for i in test_csv[trial]:
                file_list += [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith(f'R{i:02d}.edf')]

        data = []
        labels = []
        filenames = []

        with tqdm(total=len(file_list)) as pbar:
            for filename in file_list:
                eeg_data, label, n_samples = self._read_py_function(filename)
                data.append(eeg_data)
                labels.append(label)
                filenames.extend([filename] * eeg_data.shape[0])  # 对每个段重复文件名
                pbar.update(1)
        print("Loaded")
        # print("Number of samples: ", n_samples)

        data = np.vstack(data)  # 沿行堆叠数组
        labels = np.vstack(labels)
        filenames = np.array(filenames)
        # print(data.shape)
        # print(labels.shape)
        # print(filenames[0])
        # print(filenames[119:124])
        # print(filenames[130])
        # print(labels[0])
        # print(labels[119:124])
        # print(labels[130])


        return data, labels, filenames

    def _read_py_function(self, filename):
        f = pyedflib.EdfReader(filename)  # 读取数据f实例
        total_channels = f.signals_in_file

        if self.channel_indices is None:
            selected_channels = min(self.n_channels, total_channels)
            channels_to_read = range(selected_channels)
        else:
            channels_to_read = [i for i in self.channel_indices if i < total_channels]
            selected_channels = len(channels_to_read)

        eeg_data = np.zeros((selected_channels, f.getNSamples()[0]), dtype=np.float32)
        for idx, i in enumerate(channels_to_read):
            eeg_data[idx, :] = f.readSignal(i)

        n_samples = f.getNSamples()[0]

        # reminder = int(n_samples % 160)
        # n_samples -= reminder
        # seconds = int(n_samples / 160)

        reminder = int(n_samples % 500)
        n_samples -= reminder
        seconds = int(n_samples / 500)

        # print("Number of samples: ", n_samples)

        path = filename.split("\\")
        person_id = int(path[-1].partition("S")[2].partition("R")[0])
        label = np.zeros(6, dtype=bool)
        label[person_id - 1] = 1
        labels = np.tile(label, (seconds, 1))

        eeg_data = eeg_data.transpose()
        if reminder > 0:
            eeg_data = eeg_data[:-reminder, :]

        # # 归一化 EEG 数据
        # eeg_data_mean = np.mean(eeg_data, axis=0)
        # eeg_data_std = np.std(eeg_data, axis=0)
        # eeg_data = (eeg_data - eeg_data_mean) / (eeg_data_std + 1e-4)  # 防止除以零

        intervals = np.linspace(0, n_samples, num=seconds, endpoint=False, dtype=int)
        eeg_data = np.split(eeg_data, intervals)
        del eeg_data[0]
        eeg_data = np.array(eeg_data)

        return eeg_data, labels, n_samples

class PersonDataset(Dataset):
    def __init__(self, mode, personidx, trial):
        self.data, self.labels = self.load_data(mode, personidx, trial)
        self.personidx = personidx
        self.trial = trial

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].transpose((1, 0))  # 确保数据形状为 (channels, sequence_length)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

    def load_data(self, mode, personidx, trial):
        if trial == 3:
            trial = 0
        path = r"C:\Projects\person identification\eeg-motor-movementimagery-dataset-1.0.0\files"
        file_list = []
        # for i in range(1, 15):
        #     file_list += [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith(f'R{i:02d}.edf')]
        if mode == "train":
            for i in train_person_csv[trial]:
                file_list += [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith(f'R{i:02d}.edf')]
        elif mode == "test":
            for i in test_person_csv[trial]:
                file_list += [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith(f'R{i:02d}.edf')]

        target_data = []
        target_labels = []
        other_data = []
        other_labels = []
        # 添加一个列表用于记录其他数据的 person_id
        other_filename = []

        with tqdm(total=len(file_list)) as pbar:
            for filename in file_list:
                eeg_data, labels, n_samples, person_id, filenames = self._read_py_function(filename, personidx)
                # print(eeg_data.shape) # (61, 160, 64)
                # print(labels.shape) # (61, 6)

                # Check if this data belongs to the target person
                if person_id == personidx:
                    target_data.append(eeg_data)
                    # Set target labels to [1, 0]
                    target_labels.append(labels)
                else:
                    random_sample_index = random.choice(range(eeg_data.shape[0]))
                    random_sample = eeg_data[random_sample_index:random_sample_index+1]
                    random_label = labels[random_sample_index]  # 获取该样本的标签
                    other_data.append(random_sample)
                    # Set non-target labels to [0, 1]
                    other_labels.append(random_label)

                pbar.update(1)

        # Stack the target person's data and labels
        target_data = np.vstack(target_data)
        target_labels = np.vstack(target_labels)
        print(target_data.shape)

        # Calculate the sample count for target person
        n_target_samples = target_data.shape[0]
        print(f"属于此个体分类的训练样本数量：{n_target_samples}")

        # # 抽取每个非目标个体的一条数据
        # unique_filename = list(set(other_filename))  # 去重
        # sampled_other_data = []
        # sampled_other_labels = []

        # # Randomly sample n_target_samples from other_data
        # combined_other_data = np.vstack(other_data)
        # combined_other_labels = np.vstack(other_labels)
        # combined_other_person_ids = np.vstack(other_person_ids)

        # for filename in unique_filename:
        #     # 获取该 person_id 对应的所有数据索引
        #     indices = [i for i, pid in enumerate(unique_filename) if pid == filename]
        #
        #     # 随机选择一个索引
        #     selected_index = random.choice(indices)
        #
        #     # 提取对应的数据和标签
        #     sampled_other_data.append(other_data[selected_index])
        #     sampled_other_labels.append(other_labels[selected_index])

        # 将采样的其他数据堆叠
        other_data = np.vstack(other_data)
        other_labels = np.vstack(other_labels)
        print(other_data.shape)
        n_other_samples = other_labels.shape[0]
        print(f"其他个体分类的训练样本数量：{n_other_samples}")

        # other_indices = random.sample(range(len(combined_other_data)), n_target_samples)
        # sampled_other_data = combined_other_data[other_indices]
        # sampled_other_labels = combined_other_labels[other_indices]

        # Concatenate target data with sampled non-target data
        data = np.concatenate((target_data, other_data), axis=0)
        labels = np.concatenate((target_labels, other_labels), axis=0)

        return data, labels

    def _read_py_function(self, filename, personidx):
        f = pyedflib.EdfReader(filename)  # 读取数据f实例
        total_channels = f.signals_in_file

        eeg_data = np.zeros((total_channels, f.getNSamples()[0]), dtype=np.float32)
        for i in range(total_channels):
            eeg_data[i, :] = f.readSignal(i)

        n_samples = f.getNSamples()[0]
        reminder = int(n_samples % 160)
        n_samples -= reminder
        seconds = int(n_samples / 160)
        # print("Number of samples: ", n_samples)

        path = filename.split("\\")
        person_id = int(path[-1].partition("S")[2].partition("R")[0])

        task_id = int(path[-1].partition("S")[2].partition("R")[2].partition(".")[0])
        label = np.zeros(6, dtype=bool)
        if (person_id == personidx):
            if (task_id == 1 or task_id == 2):
                label[0] = 1
                label[2] = 1
            elif (task_id == 3 or task_id == 5 or task_id == 7 or task_id == 9 or task_id == 11 or task_id == 13):
                label[0] = 1
                label[3] = 1
            elif (task_id == 4 or task_id == 6 or task_id == 8 or task_id == 10 or task_id == 12 or task_id == 14):
                label[0] = 1
                label[4] = 1
        else:
            label[1] = 1
            label[5] = 1
        labels = np.tile(label, (seconds, 1))
        filenames = np.tile(filename, (seconds, 1))

        eeg_data = eeg_data.transpose()
        if reminder > 0:
            eeg_data = eeg_data[:-reminder, :]

        intervals = np.linspace(0, n_samples, num=seconds, endpoint=False, dtype=int)
        eeg_data = np.split(eeg_data, intervals)
        del eeg_data[0]
        eeg_data = np.array(eeg_data)

        return eeg_data, labels, n_samples, person_id, filenames

class ChannelDataset(Dataset):
    def __init__(self, mode, n_channels, importance_matrix, trial):
        self.n_channels = n_channels  # 新增参数
        self.importance_matrix = importance_matrix
        self.trial = trial
        self.data, self.labels, self.select_channel = self.load_data(mode, trial)

    def select_channels(self, person_row):
        # 获取person_row对象最重要的前n_channels个电极索引
        row_values = self.importance_matrix[person_row-1]
        sorted_indices = np.argsort(row_values)[-self.n_channels:]
        return np.sort(sorted_indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].transpose((1, 0))  # 确保数据形状为 (channels, sequence_length)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

    def load_data(self, mode, trial):
        if trial == 3:
            trial = 0
        path = r"C:\Projects\person identification\eeg-motor-movementimagery-dataset-1.0.0\files"
        file_list = []
        if mode == "train":
            for i in train_csv[trial]:
                file_list += [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith(f'R{i:02d}.edf')]
        elif mode == "test":
            for i in test_csv[trial]:
                file_list += [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith(f'R{i:02d}.edf')]

        data = []
        labels = []
        select_channel = []

        with tqdm(total=len(file_list)) as pbar:
            for filename in file_list:
                eeg_data, label, person_id, channel_indices = self._read_py_function(filename)
                data.append(eeg_data)
                labels.append(label)
                select_channel.append(channel_indices)
                pbar.update(1)
        print("Loaded")

        # 将 select_channel 转换为 DataFrame
        select_channel_df = pd.DataFrame(select_channel)

        # 保存为 CSV 文件
        select_channel_df.to_csv('select_channel.csv', index=False)

        print("Loaded and saved select_channel to select_channel.csv")
        # print("Number of samples: ", n_samples)

        data = np.vstack(data)  # 沿行堆叠数组
        labels = np.vstack(labels)

        return data, labels, select_channel

    def _read_py_function(self, filename):

        f = pyedflib.EdfReader(filename)  # 读取数据f实例
        total_channels = f.signals_in_file

        n_samples = f.getNSamples()[0]
        reminder = int(n_samples % 160)
        n_samples -= reminder
        seconds = int(n_samples / 160)

        path = filename.split("\\")
        person_id = int(path[-1].partition("S")[2].partition("R")[0])
        label = np.zeros(109, dtype=bool)
        label[person_id - 1] = 1
        labels = np.tile(label, (seconds, 1))

        channel_indices = self.select_channels(person_id)
        # 根据选择的电极索引读取数据
        channels_to_read = [i for i in channel_indices if i < total_channels]
        eeg_data = np.zeros((len(channels_to_read), f.getNSamples()[0]), dtype=np.float32)
        for idx, i in enumerate(channels_to_read):
            eeg_data[idx, :] = f.readSignal(i)


        eeg_data = eeg_data.transpose()
        if reminder > 0:
            eeg_data = eeg_data[:-reminder, :]

        # # 归一化 EEG 数据
        # eeg_data_mean = np.mean(eeg_data, axis=0)
        # eeg_data_std = np.std(eeg_data, axis=0)
        # eeg_data = (eeg_data - eeg_data_mean) / (eeg_data_std + 1e-4)  # 防止除以零

        intervals = np.linspace(0, n_samples, num=seconds, endpoint=False, dtype=int)
        eeg_data = np.split(eeg_data, intervals)
        del eeg_data[0]
        eeg_data = np.array(eeg_data)

        return eeg_data, labels, person_id, channel_indices