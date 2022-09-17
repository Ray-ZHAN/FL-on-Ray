import ray
import torch
# import torch.utils.data
import torch.nn.functional as F

import copy
from fl.fl_model import *
from utils.data_split import SplitedDataset

from collections import OrderedDict

@ray.remote
class ParameterServer:
	"""docstring for ParameterServer"""
	def __init__(self, test_dataset, args_batch_size):
		# super(ParameterServer, self).__init__()
		# self.arg = arg
		self.model = MNISTNet()
		self.initial_weight = copy.deepcopy(self.model.state_dict())
		self.test_data = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args_batch_size, shuffle=False)

	def model_average(self, *local_weights):
		# pass
		# local weights insist the local updates of clients in current round
		# global weight is the aggreagtion weight of fedavg
		global_weight = OrderedDict()
		cur_num_clients = len(local_weights)
		for index, local_weight in enumerate(local_weights):
			for key in self.model.state_dict().keys():
				if index == 0:
					# pass
					global_weight[key] = 1. / cur_num_clients * local_weight[key]
				else:
					global_weight[key] += 1. / cur_num_clients * local_weight[key]
		self.model.set_weight(global_weight)
		return self.model.get_weight()

	def reset_weight(self):
		# pass
		self.model.set_weight(self.initial_weight)
		return self.model.get_weight()

	# def get_global_weight(self):
	# 	# pass
	# 	# global_weight = copy.deepcopy(self.model.)
	# 	return self.model.get_weight()

	def evaluate(self):
		self.model.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(self.test_data):
				outputs = self.model(data)
				_, predicted = torch.max(outputs.data, 1)
				total += target.size(0)
				correct += (predicted == target).sum().item()
		return 100. * correct / total

@ray.remote
class DataWorker:
	"""docstring for Dataworker"""
	def __init__(self, train_dataset, train_idx, args_batch_size, args_fl_lr):
		# super(Dataworker, self).__init__()
		# self.arg = arg
		self.model = MNISTNet()
		# self.epochs = args.epochs
		self.train_data = torch.utils.data.DataLoader(dataset=SplitedDataset(train_dataset, train_idx), batch_size=args_batch_size, shuffle=True)
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args_fl_lr)

	def local_train(self, global_weight, local_epochs):
		# pass
		self.model.set_weight(global_weight)
		for index in range(local_epochs):
			# pass
			for i, (data, target) in enumerate(self.train_data):
				self.optimizer.zero_grad()
				outputs = self.model(data)
				loss = F.nll_loss(outputs, target)
				loss.backward()
				self.optimizer.step()
		return self.model.get_weight()
