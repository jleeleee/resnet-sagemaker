# The official PyTorch CNN model with MNIST training script
# https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import torchvision
from packaging.version import Version
from functools import partial
import collections
import logging
import torch._dynamo.config
# ====================================#
# 0. Import SMDebug framework class. #
# ====================================#
import smdebug.pytorch2 as smd
import os

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, loss_fn, device, train_loader, optimizer, epoch, hook):
    model.train()
    # =================================================#
    # 1. Set the SMDebug hook for the training phase. #
    # =================================================#
    if hook:
        hook.set_mode(smd.modes.TRAIN)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test(model, loss_fn, device, test_loader, hook):
    model.eval()
    # ===================================================#
    # 2. Set the SMDebug hook for the validation phase. #
    # ===================================================#
    if hook:
        hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(
                test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def create_smdebug_hook(model, loss_fn):
    out_dir="/home/ubuntu/resnet-sagemaker/pytorch/debugger_logs/"
    # delete out_dir if it already exists
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    save_config = smd.SaveConfig()
    hook = smd.Hook(
        out_dir=out_dir,
        export_tensorboard=True,
        save_config=save_config,
        include_collections=[
            'weights', 
            'biases', 
            'activations', 
            # 'losses', 
            # 'gradients',
            #'metrics'
            ]
    )
    hook.register_hook(model)
    hook.register_loss(loss_fn)
    return hook



def main():


    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--num_workers", type=int, default=1,
                        help="number of workers (GPUs)")
    parser.add_argument(
        "--dry-run", action="store_true", default=False, help="quickly check a single pass"
    )
    parser.add_argument("--seed", type=int, default=1,
                        metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )
    parser.add_argument(
        "--region", type=str, help="aws region"
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": args.num_workers,
                       "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # =======================================#
    # 3. Set data source for MNIST dataset. #
    # =======================================#

    TORCHVISION_VERSION = "0.9.1"
    if Version(torchvision.__version__) < Version(TORCHVISION_VERSION):
        # Set path to data source and include checksum to make sure data isn't corrupted
        datasets.MNIST.resources = [
            (
                f"https://sagemaker-example-files-prod-{args.region}.s3.amazonaws.com/datasets/image/MNIST/train-images-idx3-ubyte.gz",
                "f68b3c2dcbeaaa9fbdd348bbdeb94873",
            ),
            (
                f"https://sagemaker-example-files-prod-{args.region}.s3.amazonaws.com/datasets/image/MNIST/train-labels-idx1-ubyte.gz",
                "d53e105ee54ea40749a09fcbcd1e9432",
            ),
            (
                f"https://sagemaker-example-files-prod-{args.region}.s3.amazonaws.com/datasets/image/MNIST/t10k-images-idx3-ubyte.gz",
                "9fb629c4189551a2d022fa330f9573f3",
            ),
            (
                f"https://sagemaker-example-files-prod-{args.region}.s3.amazonaws.com/datasets/image/MNIST/t10k-labels-idx1-ubyte.gz",
                "ec29112dd5afa0611ce80d1b7f02629c",
            ),
        ]
    else:
        # Set path to data source
        datasets.MNIST.mirrors = [
            "https://sagemaker-sample-files.s3.amazonaws.com/datasets/image/MNIST/"]

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("/home/ubuntu/resnet-sagemaker/pytorch/mnist/data", train=True,
                              download=True, transform=transform)
    dataset2 = datasets.MNIST("/home/ubuntu/resnet-sagemaker/pytorch/mnist/data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # torch._dynamo.config.skip_nnmodule_hook_guards=False
    model = Net().to(device)
    loss_fn = nn.NLLLoss()
    hook = create_smdebug_hook(model, loss_fn)
    # hook = None


    # ======================================================#
    # 4. Register the SMDebug hook to save output tensors. #
    # ======================================================#
    # def save_activations(name, mod, inp, out):
    #     print("{}: {}".format(name, out[0].shape))
    # for name, module in model.named_modules():
    #     pname = model._get_name() + "_" + name
    #     module.register_forward_hook(partial(save_activations, pname))

    # def save_grads(pname, grad):
    #     if grad is not None:
    #         print("{}: {}".format(pname, grad.shape))
    # for name, param in model.named_parameters():
    #     pname = model._get_name() + "_" + name
    #     if param.requires_grad:
    #         param.register_hook(partial(save_grads, pname))


    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    model = torch.compile(model)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        # ===========================================================#
        # 5. Pass the SMDebug hook to the train and test functions. #
        # ===========================================================#
        train(args, model, loss_fn, device,
              train_loader, optimizer, epoch, hook)
        test(model, loss_fn, device, test_loader, hook)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    torch._functorch.debug_graphs = True
    torch._functorch.log_level = logging.DEBUG
    # torch._dynamo.config.log_level = logging.INFO
    torch._dynamo.config.print_graph_breaks = True
    main()

