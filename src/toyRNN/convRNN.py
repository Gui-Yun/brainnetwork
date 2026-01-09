import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True):
        super(ConvRNNCell, self).__init__()
        padding = kernel_size // 2
        self.input_channels = input_size
        self.hidden_channels = hidden_size
        self.conv = nn.Conv2d(in_channels=input_size + hidden_size,
                              out_channels=hidden_size,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, x, h_prev):
        # x: (Batch, C_in, H, W)
        # h_prev: (Batch, C_hidden, H, W)
        combined = torch.cat([x, h_prev], dim=1)
        h_prev = torch.tanh(self.conv(combined))
        return h_prev
    
class V1Model(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes):
        super(V1Model, self).__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvRNNCell(input_channels, hidden_channels, kernel_size=3)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, video_seq):
        # video_seq: (Batch, Time, Channels, H, W)
        batch_size, seq_len, _, h, w = video_seq.size()
        h_t = torch.zeros(batch_size, self.hidden_channels, h, w).to(video_seq.device)
        
        all_states = []
        for t in range(seq_len):
            h_t = self.cell(video_seq[:, t], h_t)
            all_states.append(h_t)
        
        # 使用最后一帧进行分类
        out = F.adaptive_avg_pool2d(h_t, (1, 1)).view(batch_size, -1)
        logits = self.classifier(out)
        
        return logits, all_states