import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from model import EEGNet, MyNet
from dataset import PersonDataset

# Assuming your EEGNet and PersonDataset classes are already defined and imported

# Parameters
# trial = 3
# num_channels = 64
# num_epochs = 250
# batch_size = 256
# learning_rate = 0.001

trial = 3
num_channels = 64
num_epochs = 100
batch_size = 64
learning_rate = 0.001

save_dir = "personmodels"
os.makedirs(save_dir, exist_ok=True)  # Directory to save models and weights

# Initialize a 109*64 matrix to store eca1 weights
eca1_weights_matrix = np.zeros((109, num_channels, num_channels))

# List to store test accuracy for each person
test_accuracies = []

for personidx in range(1, 110):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Training model for person {personidx}...")

    # Initialize the dataset for the specific person
    train_dataset = PersonDataset("train", personidx, trial)
    test_dataset = PersonDataset("test", personidx, trial)
    # print(dataset.data.shape)
    # print(dataset.labels.shape)
    # break

    # Split into training and validation sets
    # n_samples = len(dataset.data)
    # n_train = int(n_samples * 0.8)
    # n_test = n_samples - n_train
    # train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # for inputs, labels in train_loader:
    #     print(inputs[1].shape)
    #     print(labels[1].shape)
    #     print(inputs[1])
    #     print(labels[1])
    #     break
    # break

    # Initialize the model, loss function, and optimizer
    # model = EEGNet(num_channels=num_channels).to(device)
    model = MyNet(num_channels=num_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.max(1)[1])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels.max(1)[1]).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f"Person {personidx} | Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

    # Save the model for this person
    model_path = os.path.join(save_dir, f"model_person_{personidx}.pth")
    torch.save(model.state_dict(), model_path)

    # Evaluate on validation set
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_correct += (predicted == labels.max(1)[1]).sum().item()
            test_total += labels.size(0)

    val_acc = 100 * test_correct / test_total
    test_accuracies.append({"Person": personidx, "Test Accuracy": val_acc})
    print(f"Person {personidx} | Test Accuracy: {val_acc:.2f}%")

    # Extract eca1 layer weights and store in the eca1_weights_matrix
    eca1_weights = model.eca1.fc.weight.cpu().detach().numpy()  # Assuming weight is an attribute
    eca1_weights_matrix[personidx-1] = eca1_weights

# Save the eca1 weights matrix
np.save(os.path.join(save_dir, "eca1_weights_matrix.npy"), eca1_weights_matrix)
print("Training complete. ECA1 weights matrix saved.")

absolute_weights_matrix = np.abs(eca1_weights_matrix)

# 按列求和，得到一个形状为 (109, 64) 的矩阵
column_sum_matrix = np.sum(absolute_weights_matrix, axis=1)
np.save(os.path.join(save_dir, "column_sum_matrix.npy"), column_sum_matrix)
print(f"Column sum matrix saved.")

# Save test accuracies to a CSV file
csv_file_path = os.path.join(save_dir, "test_accuracies.csv")
with open(csv_file_path, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["Person", "Test Accuracy"])
    writer.writeheader()
    writer.writerows(test_accuracies)
print(f"Test accuracies saved to {csv_file_path}.")