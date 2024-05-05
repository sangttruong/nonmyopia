import torch

class MultiLayerLinearModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiLayerLinearModel, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_size*2)
        self.layer2 = torch.nn.Linear(hidden_size*2, hidden_size*2)
        self.layer3 = torch.nn.Linear(hidden_size*2, output_size, bias=False)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x