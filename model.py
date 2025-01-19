import torch
import torch.nn as nn
import torch.nn.functional as F

# class SelfAttention(nn.Module):
#     def __init__(self, input_dim):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)

    # def forward(self, x):
    #     Q = self.query(x)
    #     K = self.key(x)
    #     V = self.value(x)
    #     attention_weights = nn.functional.softmax(torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5), dim=-1)
    #     out = torch.bmm(attention_weights, V)
    #     return out

# class OrthogonalSelfAttention(nn.Module):
#     def __init__(self, input_dim):
#         super(OrthogonalSelfAttention, self).__init__()
#         self.query = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)
#
#     def forward(self, x):
#         Q = self.query(x)
#         K = self.key(x)
#         V = self.value(x)
#
#         Q_orthogonal = torch.linalg.qr(Q)[0]
#         K_orthogonal = torch.linalg.qr(K)[0]
#
#         attention_weights = nn.functional.softmax(
#             torch.bmm(Q_orthogonal, K_orthogonal.transpose(1, 2)) / (x.size(-1) ** 0.5), dim=-1)
#
#         # 输出计算
#         out = torch.bmm(attention_weights, V)
#         return out


# class CNNLSTM(nn.Module):
#     def __init__(self, n_channels, lstm_size, lstm_layers, n_classes):
#         super(CNNLSTM, self).__init__()
#
#         self.batch_norm = nn.BatchNorm1d(n_channels, eps=0.0001)
#
#         # Padding manually calculated for 'same' effect
#         padding_conv = 1  # For kernel size 2
#         padding_pool = 1  # For pool size 2
#
#         self.conv1 = nn.Conv1d(n_channels, n_channels*2, kernel_size=2, stride=1, padding=padding_conv)
#         self.pool1 = nn.MaxPool1d(2, stride=2, padding=padding_pool)
#
#         self.conv2 = nn.Conv1d(n_channels*2, n_channels*4, kernel_size=2, stride=1, padding=padding_conv)
#         self.pool2 = nn.MaxPool1d(2, stride=2, padding=padding_pool)
#
#         self.conv3 = nn.Conv1d(n_channels*4, n_channels*8, kernel_size=2, stride=1, padding=padding_conv)
#         self.pool3 = nn.MaxPool1d(2, stride=2, padding=padding_pool)
#
#         self.conv4 = nn.Conv1d(n_channels*8, n_channels*16, kernel_size=2, stride=1, padding=padding_conv)
#         self.pool4 = nn.MaxPool1d(2, stride=2, padding=padding_pool)
#
#         self.lstm_size = lstm_size
#         self.lstm_layers = lstm_layers
#
#         # self.attention = SelfAttention(n_channels * 16)  # 添加自注意力层
#
#         # LSTM layer
#         self.lstm = nn.LSTM(n_channels*16, hidden_size=lstm_size, num_layers=lstm_layers, batch_first=True,
#                             dropout=0.5)
#
#         self.fc1 = nn.Linear(lstm_size, n_channels*8)
#         self.fc2 = nn.Linear(n_channels*8, n_classes)
#
#     def forward(self, x):
#         x = self.batch_norm(x)
#         x = self.conv1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.conv3(x)
#         x = self.pool3(x)
#         x = self.conv4(x)
#         x = self.pool4(x)
#
#         # Prepare data for LSTM
#         # (batch_size, num_channels, seq_len) -> (batch_size, seq_len, num_channels)
#         x = x.permute(0, 2, 1)
#
#         # LSTM 之前添加自注意力机制
#         # x = self.attention(x)
#
#         # LSTM forward pass
#         lstm_out, _ = self.lstm(x)
#
#         # Only use the output of the last LSTM cell
#         x = lstm_out[:, -1, :]
#
#         # Fully connected layers
#         x = self.fc1(x)
#         x = self.fc2(x)
#
#         return x


class CNNTransformer(nn.Module):
    def __init__(self, n_channels, n_heads, num_layers, n_classes):
        super(CNNTransformer, self).__init__()

        self.batch_norm = nn.BatchNorm1d(n_channels, eps=0.0001)

        # Padding manually calculated for 'same' effect
        padding_conv = 1  # For kernel size 2
        padding_pool = 1  # For pool size 2

        self.conv1 = nn.Conv1d(n_channels, n_channels * 2, kernel_size=2, stride=1, padding=padding_conv)
        self.pool1 = nn.MaxPool1d(2, stride=2, padding=padding_pool)

        self.conv2 = nn.Conv1d(n_channels * 2, n_channels * 4, kernel_size=2, stride=1, padding=padding_conv)
        self.pool2 = nn.MaxPool1d(2, stride=2, padding=padding_pool)

        self.conv3 = nn.Conv1d(n_channels * 4, n_channels * 8, kernel_size=2, stride=1, padding=padding_conv)
        self.pool3 = nn.MaxPool1d(2, stride=2, padding=padding_pool)

        self.conv4 = nn.Conv1d(n_channels * 8, n_channels * 16, kernel_size=2, stride=1, padding=padding_conv)
        self.pool4 = nn.MaxPool1d(2, stride=2, padding=padding_pool)

        # Transformer configuration
        self.embedding_dim = n_channels * 16
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=n_heads, dim_feedforward=self.embedding_dim*2, dropout=0.5, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc1 = nn.Linear(self.embedding_dim, n_channels * 8)
        self.fc2 = nn.Linear(n_channels * 8, n_classes)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)

        # Prepare data for Transformer
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, embedding_dim)

        # Transformer forward pass
        x = self.transformer_encoder(x)

        # Use the output of the last time step
        x = x[:, -1, :]  # (batch_size, embedding_dim)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class ECAModuleFC(nn.Module):
    def __init__(self, num_channels):
        """
        ECA Module using a fully connected (FC) layer for channel attention.

        Args:
            num_channels (int): Number of input channels.
        """
        super(ECAModuleFC, self).__init__()
        self.fc = nn.Linear(num_channels, num_channels, bias=False)

    def forward(self, x):
        """
        Forward pass for the ECA module with FC layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width)

        Returns:
            torch.Tensor: Output tensor after applying ECA with FC layer.
        """
        # Global Average Pooling (GAP) across spatial dimensions
        y = F.adaptive_avg_pool1d(x, 1)  # Shape: (batch_size, num_channels, 1)
        # y_tanh = torch.tanh(y)
        y = y.squeeze(-1)  # Shape: (batch_size, num_channels)

        # Fully connected layer to obtain channel-wise weights
        y = self.fc(y)  # Shape: (batch_size, num_channels)

        # Sigmoid activation to obtain normalized weights
        y = torch.sigmoid(y).unsqueeze(-1)  # Shape: (batch_size, num_channels, 1)

        # Apply channel attention weights
        out = x * y
        return out

class ECAModule(nn.Module):
    def __init__(self, num_channels):
        """
        ECA Module using a fully connected (FC) layer for channel attention.

        Args:
            num_channels (int): Number of input channels.
        """
        super(ECAModule, self).__init__()
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        """
        Forward pass for the ECA module with FC layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width)

        Returns:
            torch.Tensor: Output tensor after applying ECA with FC layer.
        """
        # Global Average Pooling (GAP) across spatial dimensions
        y = F.adaptive_avg_pool1d(x, 1)  # Shape: (batch_size, num_channels, 1)
        # y_tanh = torch.tanh(y)
        y = y.squeeze(-1)  # Shape: (batch_size, num_channels)
        y = y.unsqueeze(1)
        # Fully connected layer to obtain channel-wise weights
        y = self.conv1d(y)  # Shape: (batch_size, num_channels)
        y = y.squeeze(1)
        # Sigmoid activation to obtain normalized weights
        y = torch.sigmoid(y).unsqueeze(-1)  # Shape: (batch_size, num_channels, 1)

        # Apply channel attention weights
        out = x * y
        return out


class EEGNet(nn.Module):
    def __init__(self, num_channels):
        super(EEGNet, self).__init__()
        self.eca1 = ECAModuleFC(num_channels)
        self.conv1 = nn.Conv1d(num_channels, num_channels // 2, kernel_size=2, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(2, stride=2, padding=1)
        self.eca2 = ECAModule(num_channels // 2)
        self.conv2 = nn.Conv1d(num_channels // 2, num_channels // 4, kernel_size=2, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(2, stride=2, padding=1)
        self.eca3 = ECAModule(num_channels // 4)

        # MLP: a fully connected layer with output dimension 2
        self.fc = nn.Linear(num_channels // 4, 4)

    def forward(self, x):
        # Apply the first ECA and convolution + pooling layers
        x = self.eca1(x)
        x = self.conv1(x)
        x = self.pool1(x)

        # Apply the second ECA and convolution + pooling layers
        x = self.eca2(x)
        x = self.conv2(x)
        x = self.pool2(x)

        # Apply the third ECA module
        x = self.eca3(x)

        # Flatten the output for the MLP
        x = x.mean(dim=-1)  # Global Average Pooling across the time dimension

        # Pass through the fully connected layer
        x = self.fc(x)

        return x

class MultiScaleBidirectionalFeatureBlock(nn.Module):
    def __init__(self, num_channels):
        """
        Multi-scale Bidirectional Feature Extraction Block.

        Args:
            num_channels (int): Number of input channels.
        """
        super(MultiScaleBidirectionalFeatureBlock, self).__init__()

        # Forward Local and Global Convolutions
        self.forward_local_conv = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        self.forward_global_conv = nn.Conv1d(num_channels, num_channels, kernel_size=7, padding=3)

        # Backward Local and Global Convolutions
        self.backward_local_conv = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        self.backward_global_conv = nn.Conv1d(num_channels, num_channels, kernel_size=7, padding=3)

        # Fully Connected Layers
        self.fc = nn.Linear(num_channels, num_channels)

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for Multi-scale Bidirectional Feature Extraction Block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, length)

        Returns:
            torch.Tensor: Output tensor after feature extraction.
        """
        # Process the input through the initial FC layer
        input_fc = self.fc(x.mean(dim=-1))  # Global average pooling and FC layer
        input_fc = self.sigmoid(input_fc).unsqueeze(-1)  # Add channel dimension back


        # Forward path
        forward_local = self.forward_local_conv(x)  # Local convolution
        forward_local = self.sigmoid(forward_local)
        forward_global = self.forward_global_conv(forward_local)  # Global convolution
        forward_output = self.fc(forward_global.transpose(1, 2)).transpose(1, 2)  # Fully connected
        forward_combined = forward_output + input_fc

        # Backward path
        backward_local = self.backward_local_conv(x.flip(-1))  # Local convolution
        backward_local = self.sigmoid(backward_local)
        backward_global = self.backward_global_conv(backward_local)  # Global convolution
        backward_output = self.fc(backward_global.transpose(1, 2)).transpose(1, 2)  # Fully connected
        backward_combined = backward_output + input_fc

        # Combine forward and backward paths
        combined_output = forward_combined * backward_combined

        return combined_output

class MyNet(nn.Module):
    def __init__(self, num_channels):
        super(MyNet, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_channels, eps=0.0001)
        self.eca1 = ECAModuleFC(num_channels)
        self.mbf1 = MultiScaleBidirectionalFeatureBlock(num_channels)
        self.mbf2 = MultiScaleBidirectionalFeatureBlock(num_channels)
        self.mbf3 = MultiScaleBidirectionalFeatureBlock(num_channels)

        # MLP: a fully connected layer with output dimension 2
        self.fc1 = nn.Linear(num_channels, num_channels//4)
        self.fc2 = nn.Linear(num_channels//4, 6)

    def forward(self, x):
        x = self.batch_norm(x)
        # Apply the first ECA and convolution + pooling layers
        # print("Input to ECA: ", x.shape)
        x = self.eca1(x)
        # print("After ECA: ", x.shape)

        # Apply the third ECA module
        x = self.mbf1(x)
        # print("After MBF1: ", x.shape)
        x = self.mbf2(x)
        # print("After MBF2: ", x.shape)
        x = self.mbf3(x)

        # Flatten the output for the MLP
        x = x.mean(dim=-1)  # Global Average Pooling across the time dimension
        # print("After Global Average Pooling: ", x.shape)

        # Pass through the fully connected layer
        x = self.fc1(x)
        # print("After FC1: ", x.shape)
        x = self.fc2(x)
        # print("After FC2: ", x.shape)

        return x


# ******************* 正交通道注意力 ***************************
class OrthogonalAttention(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(OrthogonalAttention, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ortho_loss = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_weights = self.attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)

        # 正交性约束：对注意力权重矩阵施加正交性惩罚
        if self.training:
            ortho_loss = torch.mean((torch.bmm(attn_weights, attn_weights.transpose(1, 2)) - torch.eye(
                attn_weights.size(1)).unsqueeze(0).to(attn_weights.device)) ** 2)
            self.ortho_loss = ortho_loss  # 存储正交损失

        src = src + self.dropout(attn_output)
        src = self.norm(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm(src)
        return src

class CNNOrTransformer(nn.Module):
    def __init__(self, n_channels, n_heads, num_layers, n_classes):
        super(CNNOrTransformer, self).__init__()

        self.batch_norm = nn.BatchNorm1d(n_channels, eps=0.0001)

        padding_conv = 1
        padding_pool = 1

        self.conv1 = nn.Conv1d(n_channels, n_channels * 2, kernel_size=2, stride=1, padding=padding_conv)
        self.pool1 = nn.MaxPool1d(2, stride=2, padding=padding_pool)

        self.conv2 = nn.Conv1d(n_channels * 2, n_channels * 4, kernel_size=2, stride=1, padding=padding_conv)
        self.pool2 = nn.MaxPool1d(2, stride=2, padding=padding_pool)

        self.conv3 = nn.Conv1d(n_channels * 4, n_channels * 8, kernel_size=2, stride=1, padding=padding_conv)
        self.pool3 = nn.MaxPool1d(2, stride=2, padding=padding_pool)

        self.conv4 = nn.Conv1d(n_channels * 8, n_channels * 16, kernel_size=2, stride=1, padding=padding_conv)
        self.pool4 = nn.MaxPool1d(2, stride=2, padding=padding_pool)

        self.embedding_dim = n_channels * 16
        self.transformer_encoder = nn.ModuleList(
            [OrthogonalAttention(self.embedding_dim, n_heads, self.embedding_dim * 2, dropout=0.5) for _ in
             range(num_layers)])

        self.fc1 = nn.Linear(self.embedding_dim, n_channels * 8)
        self.fc2 = nn.Linear(n_channels * 8, n_classes)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)

        x = x.permute(0, 2, 1)

        ortho_losses = []
        for layer in self.transformer_encoder:
            x = layer(x)
            if layer.ortho_loss is not None:
                ortho_losses.append(layer.ortho_loss)

        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.fc2(x)

        # 返回模型输出和正交损失列表
        return x



