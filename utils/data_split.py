import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import Dataset

def download_data(dataset_name):
    # dataset_name indicates which dataset to download. dataset_name = 'mnist' or 'cifar' 
    if dataset_name == 'mnist':
        # download mnist
        # print('hello')
        mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST("./data", train=True, download=False, transform=mnist_transforms)
        test_dataset = datasets.MNIST("./data", train=False, download=False, transform=mnist_transforms)
    else:
        # download cifar
        pass
    print('downloading dataset successfully')
    return train_dataset, test_dataset

def split_data(dataset, distri_mode, num_clients):
    # split the training data
    # dataset = mnist or cifar
    # distri_mode = iid or non-iid
    # num_clients, split the dataset to each client
    if distri_mode == 'iid':
        num_items = int(len(dataset) / num_clients)
        dict_clients, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_clients):
            dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_clients[i])
    else:
        # split the dataset as non-iid, each client has two labels
        num_shards, num_imgs = 2 * num_clients, len(dataset) / (2 * num_clients)
        idx_shards = [i for i in range(num_shards)]
        dict_clients = {}
        all_idxs = np.arange(len(dataset))
        labels = dataset.targets.numpy()
        
        # sort labels
        idx_labels = np.vstack((all_idxs, labels))
        idx_labels = idx_labels[:, idx_labels[1,:].argsort()]
        all_idxs = idx_labels[0,:]
        
        # divide and assign
        for i in range(num_clients):
            rand_set = set(np.random.choice(idx_shards, 2, replace=False))
            idx_shards = list(set(idx_shards) - rand_set)
            for rand in rand_set:
                dict_clients[i] = np.concatenate((dict_clients[i], all_idxs[rand*num_imgs:(rand+1)*num_imgs]), axis = 0)
                
    print('splitting dataset successfully')
    return dict_clients

class SplitedDataset(Dataset):
    def __init__(self, dataset, idxs):
        super(SplitedDataset, self).__init__()
        self.dataset = dataset
        self.idxs = list(idxs)
        
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
