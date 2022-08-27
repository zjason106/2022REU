import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc0 = nn.Linear(190, 140, dtype = float) #must specify data type as float to avoid errors
		self.fc4 = nn.Linear(140, 100, dtype = float) 
		self.fc5 = nn.Linear(100, 50, dtype = float) 
		self.fc6 = nn.Linear(50, 1, dtype = float) 


	def forward(self, x):
		x = F.relu(self.fc0(x))
		x = F.relu(self.fc4(x))
		x = F.relu(self.fc5(x))
		x = self.fc6(x)
		return x

