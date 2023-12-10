import torch
import torch.nn.functional as F
from torch import nn


# set global defaults (in this particular file) for convolutions
default_conv_kwargs = {'kernel_size': 3, 'padding': 'same', 'bias': False}

# To replicate the ~95.79%-accuracy-in-110-seconds runs, you can change the base_depth from
# 64->128, train_epochs from 12.1->90, ['ema'] epochs 10->80, cutmix_size 3->10, and cutmix_epochs 6->80
hyp = {
    'opt': {
        'scaling_factor': 1./9,
    },
    'net': {
        'whitening': {
            'kernel_size': 2,
            'num_examples': 50000,
        },
        'pad_amount': 2,
        'batch_norm_momentum': .4, # * Don't forget momentum is 1 - momentum here (due to a quirk in the original paper... >:( )
        'base_depth': 64 ## This should be a factor of 8 in some way to stay tensor core friendly
    },
}

#############################################
#            Network Components             #
#############################################

# We might be able to fuse this weight and save some memory/runtime/etc, since the fast version of the network might be able to do without somehow....
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, momentum=hyp['net']['batch_norm_momentum'], weight=False, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

# Allows us to set default arguments for the whole convolution itself.
# Having an outer class like this does add space and complexity but offers us
# a ton of freedom when it comes to hacking in unique functionality for each layer type
class Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        kwargs = {**default_conv_kwargs, **kwargs}
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs

class Linear(nn.Linear):
    def __init__(self, *args, temperature=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwargs = kwargs
        self.temperature = temperature

    def forward(self, x):
        if self.temperature is not None:
            weight = self.weight * self.temperature
        else:
            weight = self.weight
        return x @ weight.T

# can hack any changes to each convolution group that you want directly in here
class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in  = channels_in
        self.channels_out = channels_out

        self.pool1 = nn.MaxPool2d(2)
        self.conv1 = Conv(channels_in,  channels_out)
        self.conv2 = Conv(channels_out, channels_out)

        self.norm1 = BatchNorm(channels_out)
        self.norm2 = BatchNorm(channels_out)

        self.activ = nn.GELU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)

        return x

class FastGlobalMaxPooling(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Previously was chained torch.max calls.
        # requires less time than AdaptiveMax2dPooling -- about ~.3s for the entire run, in fact (which is pretty significant! :O :D :O :O <3 <3 <3 <3)
        return torch.amax(x, dim=(2,3)) # Global maximum pooling

#############################################
#          Init Helper Functions            #
#############################################

def get_patches(x, patch_shape=(3, 3), dtype=torch.float32):
    # This uses the unfold operation (https://pytorch.org/docs/stable/generated/torch.nn.functional.unfold.html?highlight=unfold#torch.nn.functional.unfold)
    # to extract a _view_ (i.e., there's no data copied here) of blocks in the input tensor. We have to do it twice -- once horizontally, once vertically. Then
    # from that, we get our kernel_size*kernel_size patches to later calculate the statistics for the whitening tensor on :D
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).to(dtype) # TODO: Annotate?

def get_whitening_parameters(patches):
    # As a high-level summary, we're basically finding the high-dimensional oval that best fits the data here.
    # We can then later use this information to map the input information to a nicely distributed sphere, where also
    # the most significant features of the inputs each have their own axis. This significantly cleans things up for the
    # rest of the neural network and speeds up training.
    n,c,h,w = patches.shape
    est_covariance = torch.cov(patches.view(n, c*h*w).t())
    # this is the same as saying we want our eigenvectors, with the specification that the 
    # matrix be an upper triangular matrix (instead of a lower-triangular matrix)
    eigenvalues, eigenvectors = torch.linalg.eigh(est_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.t().reshape(c*h*w,c,h,w).flip(0)

# Run this over the training set to calculate the patch statistics, then set the initial convolution as a non-learnable 'whitening' layer
def init_whitening_conv(layer, train_set=None, num_examples=None, previous_block_data=None, pad_amount=None, freeze=True, whiten_splits=None):
    if train_set is not None and previous_block_data is None:
        if pad_amount > 0:
            previous_block_data = train_set[:num_examples,:,pad_amount:-pad_amount,pad_amount:-pad_amount] # if it's none, we're at the beginning of our network.
        else:
            previous_block_data = train_set[:num_examples,:,:,:]

    # chunking code to save memory for smaller-memory-size (generally consumer) GPUs
    if whiten_splits is None:
         previous_block_data_split = [previous_block_data] # If we're whitening in one go, then put it in a list for simplicity to reuse the logic below
    else:
         previous_block_data_split = previous_block_data.split(whiten_splits, dim=0) # Otherwise, we split this into different chunks to keep things manageable

    eigenvalue_list, eigenvector_list = [], []
    for data_split in previous_block_data_split:
        eigenvalues, eigenvectors = get_whitening_parameters(get_patches(data_split, patch_shape=layer.weight.data.shape[2:]))
        eigenvalue_list.append(eigenvalues)
        eigenvector_list.append(eigenvectors)

    eigenvalues  = torch.stack(eigenvalue_list,  dim=0).mean(0)
    eigenvectors = torch.stack(eigenvector_list, dim=0).mean(0)
    # i believe the eigenvalues and eigenvectors come out in float32 for this because we implicitly cast it to float32 in the patches function (for numerical stability)
    set_whitening_conv(layer, eigenvalues.to(dtype=layer.weight.dtype), eigenvectors.to(dtype=layer.weight.dtype), freeze=freeze)
    data = layer(previous_block_data.to(dtype=layer.weight.dtype))
    return data

def set_whitening_conv(conv_layer, eigenvalues, eigenvectors, eps=1e-2, freeze=True):
    shape = conv_layer.weight.data.shape
    # set the first n filters of the weight data to the top n significant (sorted by importance) filters from the eigenvectors
    eigenvectors_sliced = (eigenvectors/torch.sqrt(eigenvalues+eps))[-shape[0]:, :, :, :]
    conv_layer.weight.data = torch.cat((eigenvectors_sliced, -eigenvectors_sliced), dim=0)
    ## We don't want to train this, since this is implicitly whitening over the whole dataset
    ## For more info, see David Page's original blogposts (link in the README.md as of this commit.)
    if freeze: 
        conv_layer.weight.requires_grad = False

#############################################
#            Network Definition             #
#############################################

scaler = 2.
depths = {
    'init':   round(scaler**-1*hyp['net']['base_depth']), # 32  w/ scaler at base value
    'block1': round(scaler** 0*hyp['net']['base_depth']), # 64  w/ scaler at base value
    'block2': round(scaler** 2*hyp['net']['base_depth']), # 256 w/ scaler at base value
    'block3': round(scaler** 3*hyp['net']['base_depth']), # 512 w/ scaler at base value
    'num_classes': 10
}

class SpeedyConvNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict # flexible, defined in the make_net function

    # This allows you to customize/change the execution order of the network as needed.
    def forward(self, x):
        if not self.training:
            x = torch.cat((x, torch.flip(x, (-1,))))
        x = self.net_dict['initial_block']['whiten'](x)
        x = self.net_dict['initial_block']['activation'](x)
        x = self.net_dict['conv_group_1'](x)
        x = self.net_dict['conv_group_2'](x)
        x = self.net_dict['conv_group_3'](x)
        x = self.net_dict['pooling'](x)
        x = self.net_dict['linear'](x)
        if not self.training:
            # Average the predictions from the lr-flipped inputs during eval
            orig, flipped = x.split(x.shape[0]//2, dim=0)
            x = .5 * orig + .5 * flipped
        return x

def make_net(train_images):
    whiten_conv_depth = 3*hyp['net']['whitening']['kernel_size']**2
    network_dict = nn.ModuleDict({
        'initial_block': nn.ModuleDict({
            'whiten': Conv(3, whiten_conv_depth, kernel_size=hyp['net']['whitening']['kernel_size'], padding=0),
            'activation': nn.GELU(),
        }),
        'conv_group_1': ConvGroup(2*whiten_conv_depth, depths['block1']),
        'conv_group_2': ConvGroup(depths['block1'],    depths['block2']),
        'conv_group_3': ConvGroup(depths['block2'],    depths['block3']),
        'pooling': FastGlobalMaxPooling(),
        'linear': Linear(depths['block3'], depths['num_classes'], bias=False, temperature=hyp['opt']['scaling_factor']),
    })

    net = SpeedyConvNet(network_dict).cuda()
    net = net.to(memory_format=torch.channels_last) # to appropriately use tensor cores/avoid thrash while training
    net.train()
    net.half() # Convert network to half before initializing the initial whitening layer.

    ## Initialize the whitening convolution
    with torch.no_grad():
        # Initialize the first layer to be fixed weights that whiten the expected input values of the 
        # network be on the unit hypersphere. (i.e. their...average vector length is 1.?, IIRC)
        init_whitening_conv(net.net_dict['initial_block']['whiten'],
                            train_images[torch.randperm(len(train_images), device=train_images.device)],
                            num_examples=hyp['net']['whitening']['num_examples'],
                            pad_amount=hyp['net']['pad_amount'],
                            whiten_splits=5000) ## Hardcoded for now while we figure out the optimal whitening number
                                                ## If you're running out of memory (OOM) feel free to decrease this, but
                                                ## the index lookup in the dataloader may give you some trouble depending
                                                ## upon exactly how memory-limited you are

        for layer_name in net.net_dict.keys():
            if 'conv_group' in layer_name:
                # Create an implicit residual via a dirac-initialized tensor
                dirac_weights_in = torch.nn.init.dirac_(torch.empty_like(net.net_dict[layer_name].conv1.weight))

                # Add the implicit residual to the already-initialized convolutional transition layer.
                # One can use more sophisticated initializations, but this one appeared worked best in testing.
                # What this does is brings up the features from the previous residual block virtually, so not only 
                # do we have residual information flow within each block, we have a nearly direct connection from
                # the early layers of the network to the loss function.
                std_pre, mean_pre = torch.std_mean(net.net_dict[layer_name].conv1.weight.data)
                net.net_dict[layer_name].conv1.weight.data = net.net_dict[layer_name].conv1.weight.data + dirac_weights_in 
                std_post, mean_post = torch.std_mean(net.net_dict[layer_name].conv1.weight.data)

                # Renormalize the weights to match the original initialization statistics
                net.net_dict[layer_name].conv1.weight.data.sub_(mean_post).div_(std_post).mul_(std_pre).add_(mean_pre)

                ## We do the same for the second layer in each convolution group block, since this only
                ## adds a simple multiplier to the inputs instead of the noise of a randomly-initialized
                ## convolution. This can be easily scaled down by the network, and the weights can more easily
                ## pivot in whichever direction they need to go now.
                ## The reason that I believe that this works so well is because a combination of MaxPool2d
                ## and the nn.GeLU function's positive bias encouraging values towards the nearly-linear
                ## region of the GeLU activation function at network initialization. I am not currently
                ## sure about this, however, it will require some more investigation. For now -- it works! D:
                torch.nn.init.dirac_(net.net_dict[layer_name].conv2.weight)

    return net

