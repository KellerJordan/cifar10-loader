import os
import uuid
import pickle
import argparse

from tqdm import tqdm
import numpy as np

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torchvision

from model import create_model
from quick_cifar import CifarLoader

def evaluate(model, test_loader, save_outputs=False):
    model.eval()
    stats = {
        'correct': 0,
    }
    outputs_l = []
    with torch.no_grad(), autocast():
        for batch in test_loader:
            inputs, labels = batch
            outputs = model(inputs)
            if save_outputs:
                outputs_l.append(outputs)
            pred = outputs.argmax(dim=1)
            stats['correct'] += (labels == pred).sum().item()

    if save_outputs:
        outputs = torch.cat(outputs_l).cpu().numpy()
        stats['outputs'] = outputs
    return stats

def train(args, verbose=True):

    train_aug = None if args.no_aug else dict(flip=True, translate=2, cutout=12)
    train_loader = CifarLoader('/tmp/cifar10', True, args.batch_size, train_aug)
    test_loader = CifarLoader('/tmp/cifar10', False, args.batch_size)

    n_iters = args.epochs*len(train_loader)
    lr_schedule = np.interp(np.arange(1+n_iters), [0, n_iters], [1, 0])

    model = create_model().cuda(args.gpu)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    scaler = GradScaler()
    loss_fn = CrossEntropyLoss()

    log = {'args': args.__dict__,
           'losses': []}
    it = range(args.epochs)
    if verbose:
        it = tqdm(it)
    for epoch in it:
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            log['losses'].append(loss.item())
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

    if not args.no_save_outputs:
        stats = evaluate(model, test_loader, save_outputs=True)
        log['correct'] = stats['correct']
        if verbose:
            print('correct=%d' % stats['correct'])
        log['outputs'] = stats['outputs']

    os.makedirs('./logs', exist_ok=True)
    log_path = os.path.join('./logs', str(uuid.uuid4())+'.pkl')
    with open(log_path, 'wb') as f:
        pickle.dump(log, f)
    log_path = os.path.join('./logs', str(uuid.uuid4())+'.pt')
    torch.save(log, log_path)

def main(args):
    many_runs = (args.num_runs >= 10)
    it = range(args.num_runs)
    if many_runs:
        it = tqdm(it)
    try:
        for _ in it:
            train(args, verbose=not many_runs)
    except KeyboardInterrupt:
        pass

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--batch-size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=64)
parser.add_argument('--no-aug', action='store_true')
parser.add_argument('--no-save-outputs', action='store_true')
parser.add_argument('--num-runs', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

