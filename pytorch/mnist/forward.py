import torch
import torchvision as tv

import collections
from functools import partial

transforms = tv.transforms.Compose([tv.transforms.RandomResizedCrop(224, scale=(0.5, 1.)),
                                    tv.transforms.RandomHorizontalFlip(),
                                    tv.transforms.ToTensor()]) 

# transform = tv.transforms.Compose(
#         [tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))]
#     )

# train = tv.datasets.MNIST("../data", train=True,
#                               download=True, transform=transform)

train = tv.datasets.STL10('../data', download=True, transform=transforms)
model = tv.models.resnet50()

model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=len(train.classes))

model = model.to('cuda')

dataloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=8, prefetch_factor=2, pin_memory=True)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters())
    
activations = dict() # collections.defaultdict(list)

def save_activations(name, mod, inp, out):
    activations[name] = inp

forward_handles = {}

for name, module in model.named_modules():
    forward_handles[name] = module.register_forward_hook(partial(save_activations, name))

model = torch.compile(model, mode="reduce-overhead")

for i in range(2):
    activations.clear()
    batch = next(iter(dataloader))
    optimizer.zero_grad()
    x, y = batch
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        pred = model(x.to('cuda'))
        loss = loss_func(pred, y.to('cuda'))
    loss.backward()
    optimizer.step()

print(f"Recorded Layers: {activations.keys()}\n\n")
print(f"Expected Layers: {forward_handles.keys()}")