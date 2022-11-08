# the main function for drawing learned curves along 
# the rotation trajectory.
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
from orn import upgrade_to_orn 

from rotation_dataset import crease_rot_dataset
from model import Net


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORN.PyTorch MNIST-rot Example')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--use-arf', action='store_true', default=False,
                        help='upgrade to ORN')
    parser.add_argument('--num-orientation', type=int, default=8,
                        help='num_orientation for ARFs (default: 8)')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = Net()
    if args.use_arf:
        upgrade_to_orn(model, num_orientation=8, scale_factor=8, classifier=model.fc1, 
            features=model.fc1, invariant_encoding='align', encode_after_features=False)
    print(model)
    if args.cuda:
        model.cuda()

    test_loader = crease_rot_dataset()

    def test():
        model.eval()


