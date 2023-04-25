import argparse
import torch
from torch_geometric.transforms import RandomLinkSplit

from project.link_prediction import LinkPredictor
from project.datasets.wikipedia import Wikipedia


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    valid_size = args.valid_size
    test_size = args.test_size
    nhids = args.nhids
    learning_rate = args.learning_rate
    max_epochs = args.max_epochs
    
    transform = RandomLinkSplit(num_val=valid_size, num_test=test_size)
    dataset = Wikipedia(root="./data", transform=transform)
    train_data, valid_data, test_data = dataset[0]

    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    nfeats = dataset.num_features

    assert test_size + valid_size < 1.0, "test_size + valid_size must be < 1.0"

    model = LinkPredictor(nfeats=nfeats, nhids=nhids, learning_rate=learning_rate, device=device)

    model.train()

    for _ in range(max_epochs):
        train_loss = model.training_step(train_data)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--valid_size', default=0.4, type=float, help="Validation edges size. Default: 0.4")
    parser.add_argument('--test_size', default=0.2, type=float, help="Training edges size. Default: 0.2")

    parser.add_argument('--nhids', default=16, type=int, help="Number of hidden units. Default: 16")
    parser.add_argument('--learning_rate', default=0.001, type=float, help="Learning rate. Default: 0.001")
    parser.add_argument('--max_epochs', default=2, type=int, help="Max training epochs. Default: 2")


    args = parser.parse_args()

    main(args)

