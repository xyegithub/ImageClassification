# the main function for drawing learned curves along
# the rotation trajectory.
import argparse
import torch
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from orn import upgrade_to_orn
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import seaborn
color = seaborn.color_palette()
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
    model.load_state_dict(torch.load('model_baseline.pt'))
    directory = 'baseline'
    if args.use_arf:
        upgrade_to_orn(model, num_orientation=8, scale_factor=8, classifier=model.fc1,
            features=model.fc1, invariant_encoding='align', encode_after_features=False)
        model.load_state_dict(torch.load('model_orn.pt'))
        directory = 'orn'
    print(model)
    if args.cuda:
        model.cuda()

    test_loader = crease_rot_dataset()
    with torch.no_grad():
        model.eval()
        idx = 0
        for data, target in tqdm(test_loader):
            if args.cuda:
                data, target = data.cuda().squeeze(0), target.cuda()
            output = model(data).softmax(dim = 1)
            y = output[:, target.item()].cpu().numpy()
            plt.plot(np.linspace(0, 1, y.shape[0]), y, c = color[target.item()], lw = 2)
            plt.xticks([0, 0.25, 0.5, 0.75], [r'$0$', r'$\pi$/4', r'$\pi$/2', r'$3\pi$/4'])
            legend_elements = [Patch(facecolor = color[target.item()], label = "The Neuron for Digit " + str(target.item()))]
            plt.legend(handles = legend_elements, loc = 'lower right')
            plt.ylim(0, 1.1)
            plt.savefig(directory + "/" + str(idx) + ".png", dpi=120, box_inches="tight")
            idx += 1
            plt.cla()


