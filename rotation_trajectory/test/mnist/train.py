import os
import math
import numbers
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageOps
from tqdm import tqdm
from torchvision import datasets, transforms

from orn import upgrade_to_orn
from model import Net

from rotation_dataset import RandomRotateTrain

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='ORN.PyTorch MNIST-rot Example')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--use-arf', action='store_true', default=False,
                        help='upgrade to ORN')
    parser.add_argument('--num-orientation', type=int, default=8,
                        help='num_orientation for ARFs (default: 8)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.Resize(32),
                        RandomRotateTrain((-180, 180)),
                        transforms.ToTensor(),
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False, transform=transforms.Compose([
                        transforms.Resize(32),
                        RandomRotateTrain((-180, 180)),
                        transforms.ToTensor(),
                    ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = Net()
    if args.use_arf:
        upgrade_to_orn(model, num_orientation=8, scale_factor=8, classifier=model.fc1,
            features=model.fc1, invariant_encoding='align', encode_after_features=False)
    print(model)
    if args.cuda:
        model.cuda()

    optimizer = optim.Adadelta(model.parameters())
    best_test_acc = 0.0

    def train(epoch):
        model.train()
        for data, target in tqdm(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
        print(f'[{epoch}] Train loss: {loss.item():.6f}')

    def test(epoch):
        global best_test_acc
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in tqdm(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

        test_loss /= len(test_loader)
        test_acc = 100. * correct / len(test_loader.dataset)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f'Best test accuracy: {best_test_acc:.2f}%')

        print(f'[{epoch}] Test loss: {test_loss:.6f}\tAccuracy: {test_acc:.2f}%')

    for epoch in range(args.epochs):
        train(epoch)
        with torch.no_grad():
            test(epoch)

    print(f'Best test accuracy: {best_test_acc:.2f}%')

    # save parameters
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"model_{'orn' if args.use_arf else 'baseline'}.pt")
    torch.save(model.state_dict(), save_path)
