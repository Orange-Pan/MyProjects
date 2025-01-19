import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from dataset import ChannelDataset
from model import CNNTransformer


# 设置超参数
trial = 3

importance_matrix = np.load('personmodels/column_sum_matrix.npy')
n_channels = 4
n_classes = 109
batch_size = 512

# 文件路径
model_path = f'channel_checkpoints/cldnn/trial3/4/model_20241215_131611'
result_path = f'channel_results/cldnn/trial{trial}/{n_channels}/'
os.makedirs(result_path, exist_ok=True)


# 测试函数
def test():
    # 加载模型和损失函数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = CNNLSTM(n_channels, n_channels * 3, 2, n_classes).to(device)
    model = CNNTransformer(n_channels, 4, 8, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # 加载检查点
    checkpoint = torch.load(os.path.join(model_path, 'model_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载测试数据集
    test_dataset = ChannelDataset(mode="test", n_channels=n_channels, importance_matrix=importance_matrix, trial=trial)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 加载权限文件
    permission_file = "permission_status.csv"
    permission_df = pd.read_csv(permission_file)
    permission_status = dict(zip(permission_df["person_id"], permission_df["has_permission"]))

    model.eval()
    test_loss = []
    test_acc = []
    all_labels = []
    all_predictions = []
    all_filenames = []
    access_results = []

    with torch.no_grad():
        for inputs, labels in test_loader: # DataLoader的迭代器功能
            inputs, labels = inputs.to(device), labels.to(device)

            # 计算模型输出和损失
            outputs = model(inputs)
            loss = criterion(outputs, labels.max(1)[1])

            test_loss.append(loss.item())

            # 计算预测准确率
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels.max(1)[1]).sum().item()
            accuracy = correct / labels.size(0)
            test_acc.append(accuracy)

            # while i == 0:
            #     print(inputs.shape, labels.shape)
            #     print(inputs[0].shape, labels[0].shape)
            #     print(outputs.shape, labels.shape)
            #     print(outputs[0].shape, labels[0].shape)
            #     print(outputs.shape, labels.max(1)[1].shape)
            #     print(outputs, labels.max(1)[1])
            #     print(predicted)
            #     i += 1

            # 记录真实标签和预测结果
            all_labels.extend(labels.max(1)[1].cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            # all_filenames.extend(filenames)

            # # 根据权限表查询结果
            # for fname, pred in zip(filenames, predicted.cpu().numpy()):
            #     person_id = pred + 1  # 转为从1到109的person_id
            #     permission = permission_status.get(person_id, 0)  # 默认为无权限
            #     access_results.append((fname, person_id, "获得权限" if permission else "未获得权限"))

    # 计算平均测试准确率和损失
    mean_test_acc = np.mean(test_acc)
    mean_test_loss = np.mean(test_loss)

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    # 保存测试结果
    pd.DataFrame(cm).to_csv(os.path.join(result_path, 'confusion_matrix.csv'))
    pd.DataFrame(all_labels).to_csv(os.path.join(result_path, 'labels.csv'), header=None, index=None)
    pd.DataFrame(all_predictions).to_csv(os.path.join(result_path, 'predictions.csv'), header=None, index=None)
    pd.DataFrame(test_acc).to_csv(os.path.join(result_path, 'test_acc.csv'), header=None, index=None)
    pd.DataFrame(test_loss).to_csv(os.path.join(result_path, 'test_loss.csv'), header=None, index=None)

    # 保存权限验证结果
    access_results_df = pd.DataFrame(access_results, columns=["Filename", "Predicted Person ID", "Access Status"])
    access_results_df.to_csv(os.path.join(result_path, 'access_results.csv'), index=False)

    mean_metrics = {
        "Mean Test Accuracy": [mean_test_acc],
        "Mean Test Loss": [mean_test_loss]
    }

    # 保存均值结果到 CSV
    mean_metrics_df = pd.DataFrame(mean_metrics)
    mean_metrics_df.to_csv(os.path.join(result_path, 'mean_metrics.csv'), index=False)


    print(f"Mean test accuracy: {mean_test_acc:.6f}")
    print(f"Mean test loss: {mean_test_loss:.6f}")
    print("Confusion Matrix:\n", cm)

def main():
    test()


if __name__ == '__main__':
    main()