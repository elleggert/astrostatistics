from torch import nn
import torch.nn.functional as F



class LinearRegression(nn.Module):
    def __init__(self, n_input_vars = 17, n_output_vars=1):
        super().__init__() # call constructor of superclass
        self.linear = nn.Linear(n_input_vars, n_output_vars)

    def forward(self, x):
        return self.linear(x)


device = 'cpu'
model = LinearRegression().to(device)



class BaseNet(nn.Module):
    def __init__(self, n_feature = 16, n_hidden = 10, n_output = 1):
        super(BaseNet, self).__init__()
        self.fc1 = nn.Linear(n_feature,n_hidden)
        #self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = self.predict(out)
        return out

device = 'cpu'

