import math
import copy

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

from quick_cifar import CifarLoader
from model import make_net


batch_size = 1024
train_epochs = 12.1

bias_scaler = 64
# To replicate the ~95.79%-accuracy-in-110-seconds runs, you can change the base_depth from
# 64->128, train_epochs from 12.1->90, ['ema'] epochs 10->80, cutmix_size 3->10, and cutmix_epochs 6->80
hyp = {
    'opt': {
        'bias_lr':        1.525 * bias_scaler/512, # TODO: Is there maybe a better way to express the bias and batchnorm scaling? :'))))
        'non_bias_lr':    1.525 / 512,
        'bias_decay':     6.687e-4 * batch_size/bias_scaler,
        'non_bias_decay': 6.687e-4 * batch_size,
        'scaling_factor': 1./9,
        'percent_start': .23,
        'loss_scale_scaler': 1./32, # * Regularizer inside the loss summing (range: ~1/512 - 16+). FP8 should help with this somewhat too, whenever it comes out. :)
    },
    'misc': {
        'ema': {
            'epochs': 10, # Slight bug in that this counts only full epochs and then additionally runs the EMA for any fractional epochs at the end too
            'decay_base': .95,
            'decay_pow': 3.,
            'every_n_steps': 5,
        },
    }
}

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            correct += (outputs.argmax(-1) == labels).sum().item()
    return correct

class NetworkEMA(nn.Module):
    def __init__(self, net):
        super().__init__() # init the parent module so this module is registered properly
        self.net_ema = copy.deepcopy(net).eval().requires_grad_(False) # copy the model

    def update(self, current_net, decay):
        with torch.no_grad():
            for ema_net_parameter, (parameter_name, incoming_net_parameter) in zip(self.net_ema.state_dict().values(), current_net.state_dict().items()):
                if incoming_net_parameter.dtype in (torch.half, torch.float):
                    ema_net_parameter.lerp_(incoming_net_parameter.detach(), 1 - decay) # linear interpolation
                    # And then we also copy the parameters back to the network, similarly to the Lookahead optimizer (but with a much more aggressive-at-the-end schedule)
                    #if not ('norm' in parameter_name and 'weight' in parameter_name) and not 'whiten' in parameter_name:
                    if not 'whiten' in parameter_name:
                        incoming_net_parameter.copy_(ema_net_parameter.detach())

    def forward(self, inputs):
        return self.net_ema(inputs)

def init_split_parameter_dictionaries(network):
    params_non_bias = {'params': [], 'lr': hyp['opt']['non_bias_lr'], 'momentum': .85, 'nesterov': True, 'weight_decay': hyp['opt']['non_bias_decay'], 'foreach': True}
    params_bias     = {'params': [], 'lr': hyp['opt']['bias_lr'],     'momentum': .85, 'nesterov': True, 'weight_decay': hyp['opt']['bias_decay'], 'foreach': True}

    for name, p in network.named_parameters():
        if p.requires_grad:
            if 'bias' in name:
                params_bias['params'].append(p)
            else:
                params_non_bias['params'].append(p)
    return params_non_bias, params_bias

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2, reduction='none')

def main():
    train_aug = dict(flip=True, translate=2)
    train_loader = CifarLoader('/tmp/cifar10', train=True, batch_size=batch_size,
                               aug=train_aug)
    train_images = train_loader.normalize(train_loader.images)
    test_loader = CifarLoader('/tmp/cifar10', train=False, batch_size=2000)
    
    net_ema = None
    total_time_seconds = 0.
    current_steps = 0.

    num_steps_per_epoch      = len(train_loader)
    total_train_steps        = math.ceil(num_steps_per_epoch * train_epochs)
    ema_epoch_start          = math.floor(train_epochs) - hyp['misc']['ema']['epochs']

    ## I believe this wasn't logged, but the EMA update power is adjusted by being raised to the power of the number of "every n" steps
    ## to somewhat accomodate for whatever the expected information intake rate is. The tradeoff I believe, though, is that this is to some degree noisier as we
    ## are intaking fewer samples of our distribution-over-time, with a higher individual weight each. This can be good or bad depending upon what we want.
    projected_ema_decay_val  = hyp['misc']['ema']['decay_base'] ** hyp['misc']['ema']['every_n_steps']

    # Adjust pct_start based upon how many epochs we need to finetune the ema at a low lr for
    pct_start = hyp['opt']['percent_start'] #* (total_train_steps/(total_train_steps - num_low_lr_steps_for_ema))

    net = make_net(train_images)
    non_bias_params, bias_params = init_split_parameter_dictionaries(net)

    # One optimizer for the regular network, and one for the biases. This allows us to use the superconvergence onecycle training policy for our networks....
    opt = torch.optim.SGD(**non_bias_params)
    opt_bias = torch.optim.SGD(**bias_params)

    ## Not the most intuitive, but this basically takes us from ~0 to max_lr at the point pct_start, then down to .1 * max_lr at the end (since 1e16 * 1e-15 = .1 --
    ##   This quirk is because the final lr value is calculated from the starting lr value and not from the maximum lr value set during training)
    initial_div_factor = 1e16 # basically to make the initial lr ~0 or so :D
    final_lr_ratio = .07 # Actually pretty important, apparently!
    lr_sched      = OneCycleLR(opt, max_lr=non_bias_params['lr'], pct_start=pct_start,
                               div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio),
                               total_steps=total_train_steps, anneal_strategy='linear', cycle_momentum=False)
    lr_sched_bias = OneCycleLR(opt_bias, max_lr=bias_params['lr'], pct_start=pct_start,
                               div_factor=initial_div_factor, final_div_factor=1./(initial_div_factor*final_lr_ratio),
                               total_steps=total_train_steps, anneal_strategy='linear', cycle_momentum=False)

    ## For accurately timing GPU code
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize() ## clean up any pre-net setup operations

    for epoch in tqdm(range(math.ceil(train_epochs))):
        #################
        # Training Mode #
        #################
        torch.cuda.synchronize()
        starter.record()
        net.train()

        loss_train = None
        accuracy_train = None

        for inputs, labels in train_loader:
            outputs = net(inputs)

            loss_batchsize_scaler = 512/batch_size
            loss = loss_fn(outputs, labels).mul(hyp['opt']['loss_scale_scaler']*loss_batchsize_scaler).sum().div(hyp['opt']['loss_scale_scaler'])

            loss.backward()

            opt.step()
            opt_bias.step()

            lr_sched.step()
            lr_sched_bias.step()

            opt.zero_grad(set_to_none=True)
            opt_bias.zero_grad(set_to_none=True)

            if epoch >= ema_epoch_start and (current_steps+1) % hyp['misc']['ema']['every_n_steps'] == 0:          
                ## Initialize the ema from the network at this point in time if it does not already exist.... :D
                if net_ema is None: # don't snapshot the network yet if so!
                    net_ema = NetworkEMA(net)
                else:
                    # We warm up our ema's decay/momentum value over training exponentially according to the hyp config dictionary (this lets us move fast, then average strongly at the end).
                    net_ema.update(net, decay=projected_ema_decay_val*(current_steps/total_train_steps)**hyp['misc']['ema']['decay_pow'])

            current_steps += 1
            if current_steps == total_train_steps:
                break

        ender.record()
        torch.cuda.synchronize()
        total_time_seconds += 1e-3 * starter.elapsed_time(ender)
        
    print(total_time_seconds, evaluate(net, test_loader), evaluate(net_ema, test_loader))
    return evaluate(net_ema, test_loader)/len(test_loader.images)
    

if __name__ == "__main__":
    acc_list = []
    for run_num in range(10):
        acc_list.append(torch.tensor(main()))
    print("Mean and variance:", (torch.mean(torch.stack(acc_list)).item(), torch.std(torch.stack(acc_list)).item()))

