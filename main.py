import os
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import customDataset
import trainingFunctions as t

from pylab import zeros, arange, subplots, plt, savefig

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

training_id = 'resnet101_BCE'
dataset = '../../ssd2/iMaterialistFashion' # Path to dataset
split_train = '/anns/train'
split_val =  '/anns/validation'
arch = 'resnet101'
workers = 12 # Num of data loading workers
epochs = 150
start_epoch = 0 # Useful on restarts
batch_size = 256 #256 # Batch size
lr = 0.01 # Initial learning rate # Default 0.1, but people report better performance with 0.01 and 0.001
decay_every = 12 # Decay lr by a factor of 10 every decay_every epochs
momentum = 0.9
weight_decay = 1e-4
print_freq = 500
resume = None # Path to checkpoint top resume training
pretrained = True
# evaluate = False # Evaluate model on validation set at start
plot = True
best_prec1 = 0

# create model
if pretrained:
    print("=> using pre-trained model '{}'".format(arch))
    model = models.__dict__[arch](pretrained=True)
else:
    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch]()

# Edit model
# for param in model.parameters():
#     param.requires_grad = False # This would froze all net
# Replace the last fully-connected layer
# Parameters of newly constructed modules have requires_grad=True by default
num_classes = 228
del(model._modules['fc'])
model.fc = nn.Linear(2048,num_classes)
print(model)

if arch.startswith('alexnet') or arch.startswith('vgg'):
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
else:
    model = torch.nn.DataParallel(model,device_ids=[0, 1, 2, 3]).cuda()

# define loss function (criterion) and optimizer
# criterion = nn.CrossEntropyLoss().cuda()
# criterion = nn.MultiLabelSoftMarginLoss().cuda() # This is not the loss I want
criterion = nn.BCEWithLogitsLoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

# optionally resume from a checkpoint
if resume:
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

cudnn.benchmark = True

# Data loading code
train_dataset = customDataset.CustomDataset(
    dataset,split_train,Rescale=0,RandomCrop=224,Mirror=True)

val_dataset = customDataset.CustomDataset(
    dataset, split_val,Rescale=224,RandomCrop=0,Mirror=False)

# if distributed:
#     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
# else:
train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    num_workers=workers, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

plot_data = {}
plot_data['train_loss'] = zeros(epochs)
plot_data['train_top1'] = zeros(epochs)
plot_data['train_top5'] = zeros(epochs)
plot_data['val_loss'] = zeros(epochs)
plot_data['val_top1'] = zeros(epochs)
plot_data['val_top5'] = zeros(epochs)
plot_data['epoch'] = 0

# if evaluate:
#     t.validate(val_loader, model, criterion, print_freq, plot_data)

it_axes = arange(epochs)

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('epoch')
ax1.set_ylabel('train loss (r), val loss (y)')
ax2.set_ylabel('train TOP1 (b), val TOP1 (g), train TOP-5 (c), val TOP-5 (k)')
ax2.set_autoscaley_on(False)
ax1.set_ylim([0, 0.1])
ax2.set_ylim([0, 60])


for epoch in range(start_epoch, epochs):
    plot_data['epoch'] = epoch
    lr = t.adjust_learning_rate(optimizer, epoch, lr, decay_every)

    # train for one epoch
    plot_data = t.train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data)

    # evaluate on validation set
    plot_data, prec1 = t.validate(val_loader, model, criterion, print_freq, plot_data)

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    t.save_checkpoint(dataset, {
        'model': model,
        'epoch': epoch,
        'arch': arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, is_best, filename = dataset +'/models/CNN/' + training_id + '_epoch_' + str(epoch) + '.pth.tar')

    if plot:
        ax1.plot(it_axes[0:epoch], plot_data['train_loss'][0:epoch], 'r')
        ax2.plot(it_axes[0:epoch], plot_data['train_top1'][0:epoch], 'b')
        ax2.plot(it_axes[0:epoch], plot_data['train_top5'][0:epoch], 'c')

        ax1.plot(it_axes[0:epoch], plot_data['val_loss'][0:epoch], 'y')
        ax2.plot(it_axes[0:epoch], plot_data['val_top1'][0:epoch], 'g')
        ax2.plot(it_axes[0:epoch], plot_data['val_top5'][0:epoch], 'k')

        plt.title(training_id)
        plt.ion()
        plt.grid(True)
        plt.show()
        plt.pause(0.001)
        title = dataset +'/models/training/' + training_id + '_epoch_' + str(epoch) + '.png'  # Save graph to disk
        savefig(title, bbox_inches='tight')


