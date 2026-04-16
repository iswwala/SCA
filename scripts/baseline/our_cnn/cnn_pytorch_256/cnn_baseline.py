import torch
import torch.nn as nn
import torch.nn.functional as F

class ASCADCNN(nn.Module):
    def __init__(self, num_classes=256, input_length=700):
        super(ASCADCNN, self).__init__()
        
        # 5层卷积，匹配官方TensorFlow实现
        # Block 1
        self.conv1 = nn.Conv1d(1, 64, kernel_size=11, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=11, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=11, padding='same')
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv4 = nn.Conv1d(256, 512, kernel_size=11, padding='same')
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Block 5
        self.conv5 = nn.Conv1d(512, 512, kernel_size=11, padding='same')
        self.bn5 = nn.BatchNorm1d(512)
        self.pool5 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # 计算展平后的维度
        self.fc_input_dim = self._calculate_fc_dim(input_length)
        
        # 全连接层
        self.fc1 = nn.Linear(self.fc_input_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def _calculate_fc_dim(self, input_length):
        """计算全连接层输入维度"""
        with torch.no_grad():
            x = torch.zeros(1, 1, input_length)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = self.pool3(F.relu(self.conv3(x)))
            x = self.pool4(F.relu(self.conv4(x)))
            x = self.pool5(F.relu(self.conv5(x)))
            return x.view(1, -1).size(1)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Block 4
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        
        # Block 5
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        
        # 展平
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x