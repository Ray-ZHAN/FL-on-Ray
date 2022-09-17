import argparse

def args_parser():
    parser = argparse.ArgumentParser(description="Federated training", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument('--num_workers', default=5, type=int, metavar='N', help='number of clients in group')
    parser.add_argument('--num_workers_per_round', default=3, type=int, help='number of workers are selected in each round')
    parser.add_argument('--dataset', default='mnist', type=str, metavar='D', help='the model and dataset to be train')
    parser.add_argument('--fl_lr', default=1e-3,type=float,metavar='L1', help='learning rate of federated learning')
    parser.add_argument('--iid', default='iid', type=str, metavar='I', help='whether split data as iid form')
    parser.add_argument('--batch_size', default=100,type=int,metavar='B',help='batch size of trainset')
    parser.add_argument('--round', default=10, type=int, metavar='R', help='the value of training round')
    parser.add_argument('--epochs', default=1, type=int, metavar='E', help='the value of training epoch')
    args = parser.parse_args()
    return args