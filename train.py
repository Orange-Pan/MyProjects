import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import EEGDataset
# from model import CNNLSTM
from model import CNNTransformer, CNNOrTransformer
import os
import numpy as np
from datetime import datetime
import time
from torchsummary import summary

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

# Hyperparameters and settings
trial = 3
# channel_indices = [31, 35, 48, 52] # 4电极
# channel_indices = [10, 21, 22, 23, 29, 37, 40, 41] # 8电极

# channel_indices = [21, 22, 23, 29, 37]
# channel_indices = [40, 42]
# channel_indices = [41, 43]

# channel_indices = list(range(64))
# channel_indices = list(range(32))
channel_indices = [1, 10, 20, 30]

# channel_indices = [40, 41, 42, 43]
# channel_indices = [8, 10, 12, 21, 23, 29, 31, 35, 37, 40, 41, 46, 48, 50, 52, 54] # 16电极
# channel_indices = [0, 2, 3, 4, 6, 8, 10, 12, 14, 16, 17, 18, 20, 21, 22, 23, 26, 29, 31, 33, 35, 37, 40, 41, 46, 48, 50, 52, 54, 57, 60, 62] # 32电极

n_channels = len(channel_indices)
lstm_size = n_channels * 3
lstm_layers = 2
# batch_size = 512
batch_size = 80
epochs = 1000
n_classes = 6
learning_rate = 0.0001
save_path = f'checkpoints/cldnn/trial{trial}/{n_channels}/'
checkpoint_model = f'checkpoints/cldnn/trial{trial}/{n_channels}/model_{current_time}'
os.makedirs(checkpoint_model, exist_ok=True)  # 如果文件夹已存在则不会报错

# checkpoint_path = os.path.join(checkpoint_model, 'model_checkpoint.pth')
checkpoint_early = os.path.join(checkpoint_model, 'model_early.pth')
checkpoint_final = os.path.join(save_path, 'model_final.pth')
checkpoint_best = os.path.join(checkpoint_model, 'model_best.pth')

stop_loss_threshold = 1e-7  # early stop loss threshold

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize TensorBoard
writer = SummaryWriter(save_path + 'train_accuracy/')

# Training function
def train(model, train_loader, criterion, optimizer, start_epoch=0, start_index=0, num_epochs=epochs):
    train_losses = []
    train_accuracies = []
    total_samples = len(train_loader.dataset)
    index = start_index  # continue from saved index if available
    epoch = start_epoch  # continue from saved epoch if available
    best_accuracy = 0  # Initialize the best accuracy

    try:
        while epoch < num_epochs:
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            start_time = time.time()  # Start time for epoch

            # # Adjust learning rate dynamically
            # lr = learning_rate * (0.9 ** (epoch // 10))
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr
            lr = learning_rate

            # # Set dropout probability (can modify it dynamically as well)
            # model.set_dropout_prob(keep_prob)

            for i, (inputs, labels, _) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels.max(1)[1])
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
                optimizer.step()

                # Update metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.max(1)[1]).sum().item()
                index += 1

                # Write to TensorBoard
                # writer.add_scalar('Loss/train', loss.item(), epoch)
                # writer.add_scalar('Accuracy/train', 100 * correct / total, epoch)

                # Check early stopping condition
                if loss.item() < stop_loss_threshold and correct / total == 1.0:
                    print(f"Epoch: {epoch + 1}/{num_epochs}, "
                          f"Iteration: {index}, "
                          f"Train loss: {loss.item():.10f}, "
                          f"Train acc: {100 * correct / total:.2f}%")
                    save_checkpoint(model, optimizer, epoch, index, checkpoint_early)
                    print(f"Checkpoint_early saved at epoch {epoch}")
                    break

                # Save model and print every 10 iterations
                # if epoch % 10 == 0:
                #     print(f"Epoch: {epoch + 1}/{num_epochs}, "
                #           # f"Iteration: {index}, "
                #           f"Train loss: {loss.item():.10f}, "
                #           f"Train acc: {100 * correct / total:.2f}%")
                    # save_checkpoint(model, optimizer, epoch, index, checkpoint_path)

            # End of epoch
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            # writer.add_scalar('Learning_rate', lr, epoch)

            # End time for epoch and GPU memory usage
            end_time = time.time()
            epoch_time = end_time - start_time
            gpu_memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
            gpu_memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # Convert to MB

            print(f"Epoch {epoch + 1}/{num_epochs} | Training Loss: {train_loss:.4f} | "
                  f"Training Accuracy: {train_acc:.2f}% | Epoch Time: {epoch_time:.2f}s | "
                  f"GPU Memory Allocated: {gpu_memory_allocated:.2f} MB | "
                  f"GPU Memory Reserved: {gpu_memory_reserved:.2f} MB")

            # Check if current accuracy is the best so far, if so, save the model
            if train_acc > best_accuracy:
                best_accuracy = train_acc
                save_checkpoint(model, optimizer, epoch, index, checkpoint_best)
                print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

            epoch += 1  # manually increment epoch

    except Exception as e:
        print(f"Training interrupted: {e}")
    finally:
        save_checkpoint(model, optimizer, epoch, index, checkpoint_final)
        print("Training completed or interrupted; saving final model.")
        writer.close()

def save_checkpoint(model, optimizer, epoch, index, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'index': index
    }
    torch.save(checkpoint, path)
    # print(f"Checkpoint saved at epoch {epoch}, index {index}")

def load_checkpoint(model, optimizer, path):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        index = checkpoint['index']
        print(f"Checkpoint loaded: starting at epoch {epoch}, index {index}")
        return epoch, index
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0, 0  # Start from scratch if no checkpoint

# Main function
def main():
    # Initialize model, criterion, optimizer
    # model = CNNLSTM(n_channels, lstm_size, lstm_layers, n_classes).to(device)
    model = CNNTransformer(n_channels, 4, 8, n_classes).to(device)
    # model = CNNOrTransformer(n_channels, 4, 8, n_classes).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Print model architecture
    print("Model Architecture:")
    print(model)
    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    # Print model summary (requires input size for each batch)
    # summary(model, (batch_size, n_channels, 160))  # 假设输入数据大小为 (channels, sequence_length)

    # Print other training settings
    print(f"\nTraining settings:")
    print(f"Number of epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Channel indices: {channel_indices}")
    print(f"Device: {device}")


    # Load dataset and create DataLoader
    train_dataset = EEGDataset(mode="train", n_channels=n_channels, channel_indices=channel_indices, trial=trial)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load checkpoint if available
    start_epoch, start_index = load_checkpoint(model, optimizer, checkpoint_final)

    # Start training
    train(model, train_loader, criterion, optimizer, start_epoch, start_index)

if __name__ == '__main__':
    main()
