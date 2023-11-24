import argparse
import os
import torch
import string
import random
import numpy as np
from metric import print_f_score
from dataset import PasswordDataset
from model import ContextModel
from model import PasswordModel
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.manual_seed(2021)
random.seed(2021)
np.random.seed(2021)

parser = argparse.ArgumentParser(description='Password/None-Password Classifier')

parser.add_argument('--task', choices=['context', 'password', "password_eval"], dest='task',
                    help='choose training task from: context and password')
parser.add_argument('--train_path', metavar='DIR',
                    help='path to training data csv [default: data/train.csv]',
                    default='data/train.csv')
parser.add_argument('--val_path', metavar='DIR',
                    help='path to validation data csv [default: data/test.csv]',
                    default='data/test.csv')

learn = parser.add_argument_group('Learning options')
learn.add_argument('--lr', type=float, default=0.0001, help='initial learning rate [default: 0.0001]')
learn.add_argument('--epochs', type=int, default=32, help='number of epochs for training [default: 32]')
learn.add_argument('--batch_size', type=int, default=128, help='batch size for training [default: 128]')
learn.add_argument('--max_norm', type=int, default=400, help='Norm cutoff to prevent explosion of gradients')
learn.add_argument('--optimizer', default='Adam', help='Type of optimizer. SGD|Adam|ASGD are supported [default: Adam]')
learn.add_argument('--class_weight', default=None, action='store_true',
                   help='Weights should be a 1D Tensor assigning weight to each of the classes.')
learn.add_argument('--dynamic_lr', action='store_true', default=False, help='Use dynamic learning schedule.')
learn.add_argument('--milestones', nargs='+', type=int, default=[5, 10, 15],
                   help='List of epoch indices. Must be increasing. Default:[5, 10, 15]')
learn.add_argument('--decay_factor', default=0.5, type=float,
                   help='Decay factor for reducing learning rate [default: 0.5]')

# CNN model
cnn = parser.add_argument_group('CNN Model options')
cnn.add_argument('--max_length', type=int, default=30, help='Maximum length of input sequence to CNNs [default: 30]')
cnn.add_argument('--shuffle', action='store_true', default=False, help='shuffle the data every epoch')
cnn.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
cnn.add_argument('--kernel_sum', type=int, default=100, help='number of each kind of kernel')
cnn.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')

# device
device = parser.add_argument_group('Device options')
device.add_argument('--num_workers', default=1, type=int, help='Number of workers used in data-loading')
device.add_argument('--cuda', action='store_true', default=False, help='enable the gpu')

# experiment options
experiment = parser.add_argument_group('Experiment options')
experiment.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='Turn on progress tracking per iteration for debugging')
experiment.add_argument('--continue_from', default='', help='Continue from checkpoint model')
experiment.add_argument('--checkpoint', dest='checkpoint', default=False, action='store_true',
                        help='Enables checkpoint saving of model')
experiment.add_argument('--checkpoint_per_batch', default=1000000, type=int,
                        help='Save checkpoint per batch, 0 means never save [default: 1000000]')
experiment.add_argument('--save_folder', default='models_CNN',
                        help='Locations to save epoch models, training configurations and results.')
experiment.add_argument('--log_config', default=True, action='store_true', help='Store experiment configuration')
experiment.add_argument('--log_result', default=True, action='store_true', help='Store experiment result')
experiment.add_argument('--log_interval', type=int, default=1,
                        help='how many steps to wait before logging training status [default: 1]')
experiment.add_argument('--val_interval', type=int, default=1000000,
                        help='how many steps to wait before validation [default: 1000000]')
experiment.add_argument('--save_interval', type=int, default=1,
                        help='how many epochs to wait before saving [default:1]')


def train(train_loader, dev_loader, model, args):
    # optimization scheme
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == 'ASGD':
        optimizer = optim.ASGD(model.parameters(), lr=args.lr)

    # continue training from checkpoint model
    if args.continue_from:
        print("=> loading checkpoint from '{}'".format(args.continue_from))
        assert os.path.isfile(args.continue_from), "=> no checkpoint found at '{}'".format(args.continue_from)
        checkpoint = torch.load(args.continue_from)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint.get('iter', None)
        best_acc = checkpoint.get('best_acc', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 1
        else:
            start_iter += 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 1
        start_iter = 1
        best_acc = None

    # dynamic learning scheme
    if args.dynamic_lr and args.optimizer != 'Adam':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.decay_factor,
                                                   last_epoch=-1)

    # gpu
    if args.cuda:
        model = model.cuda()
        # model = torch.nn.DataParallel(model).cuda()

    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        if args.dynamic_lr and args.optimizer != 'Adam':
            scheduler.step()

        for i_batch, data in enumerate(train_loader, start=start_iter):
            inputs, target = data

            if args.cuda:
                inputs, target = inputs.cuda(), target.cuda()

            logit = model(inputs)
            loss = F.nll_loss(logit, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            optimizer.step()

            if args.cuda:
                torch.cuda.synchronize()

            if args.verbose:
                print('\nTargets, Predicates')
                print(torch.cat(
                    (target.unsqueeze(1), torch.unsqueeze(torch.max(logit, 1)[1].view(target.size()).data, 1)), 1))
                print('\nLogit')
                print(logit)

            if i_batch % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / args.batch_size
                print('Epoch[{}] Batch[{}] - loss: {:.6f}  lr:{:.5f}  acc:{:.3f}% ({}/{})'.format(epoch,
                                                                                                  i_batch,
                                                                                                  loss.data,
                                                                                                  optimizer.state_dict()[
                                                                                                      'param_groups'][
                                                                                                      0]['lr'],
                                                                                                  accuracy,
                                                                                                  corrects,
                                                                                                  args.batch_size
                                                                                                  ))

            # if i_batch % args.val_interval == 0:
            #     val_loss, val_acc = eval(dev_loader, model, epoch, i_batch, optimizer, args)

        if args.checkpoint and epoch % args.save_interval == 0:
            file_path = '%s/CNN_epoch_%d.pth.tar' % (args.save_folder, epoch)
            print('\r=> saving checkpoint model to %s' % file_path)
            save_checkpoint(model, {'epoch': epoch,
                                    'optimizer': optimizer.state_dict(),
                                    'best_acc': best_acc}, file_path)

        # validation
        var_loss, val_acc = eval(dev_loader, model, epoch, i_batch, optimizer, args)
        if best_acc is None or val_acc > best_acc:
            file_path = '%s/model_best.pth.tar' % (args.save_folder)
            print('\r=> found better validated model, saving to %s' % file_path)
            save_checkpoint(model,
                            {'epoch': epoch,
                             'optimizer': optimizer.state_dict(),
                             'best_acc': best_acc},
                            file_path)
            best_acc = val_acc


def eval(data_loader, model, epoch_train, batch_train, optimizer, args):
    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    for i_batch, (data) in enumerate(data_loader):
        inputs, target = data
        size += len(target)
        if args.cuda:
            inputs, target = inputs.cuda(), target.cuda()
        logit = model(inputs)
        predicates = torch.max(logit, 1)[1].view(target.size()).data
        # accumulated_loss += F.nll_loss(logit, target, size_average=False).data
        accumulated_loss += F.nll_loss(logit, target, size_average=False).data
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        predicates_all += predicates.cpu().numpy().tolist()
        target_all += target.data.cpu().numpy().tolist()
        if args.cuda:
            torch.cuda.synchronize()

    avg_loss = accumulated_loss / size
    accuracy = 100.0 * corrects / size
    model.train()
    print('\nEvaluation - loss: {:.6f}  lr: {:.5f}  acc: {:.3f}% ({}/{}) '.format(avg_loss,
                                                                                  optimizer.state_dict()[
                                                                                      'param_groups'][0]['lr'],
                                                                                  accuracy,
                                                                                  corrects,
                                                                                  size))
    print_f_score(predicates_all, target_all)
    print('\n')
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
            r.write('\n{:d},{:d},{:.5f},{:.2f},{:f}'.format(epoch_train,
                                                            batch_train,
                                                            avg_loss,
                                                            accuracy,
                                                            optimizer.state_dict()['param_groups'][0]['lr']))

    return avg_loss, accuracy


def save_checkpoint(model, state, filename):
    state['state_dict'] = model.state_dict()
    torch.save(state, filename)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loader(dataset_path, max_length, batch_size, num_workers, alphabet):
    g = torch.Generator()
    g.manual_seed(2021)
    print('\nLoading data from {}'.format(dataset_path))
    dataset = PasswordDataset(dataset_path, alphabet, max_length)
    dataset_loader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                drop_last=True,
                                shuffle=True,
                                worker_init_fn=seed_worker,
                                generator=g)
    return dataset, dataset_loader


def do_train(args):
    if args.task == "context":
        alphabet = string.printable
        model = ContextModel(len(alphabet))
        max_length = args.max_length
    else:
        alphabet = string.ascii_letters + string.digits + string.punctuation
        model = PasswordModel(len(alphabet))
        max_length = 30

    train_dataset, train_loader = make_data_loader(args.train_path,
                                                   max_length,
                                                   args.batch_size,
                                                   args.num_workers,
                                                   alphabet)

    dev_dataset, dev_loader = make_data_loader(args.val_path,
                                               max_length,
                                               args.batch_size,
                                               args.num_workers,
                                               alphabet)

    # make save folder
    try:
        if not os.path.isfile(args.save_folder):
            os.makedirs(args.save_folder)
    except OSError as e:
        print(e)

    # configuration
    print("\nConfiguration:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}:".format(attr.capitalize().replace('_', ' ')).ljust(25) + "{}".format(value))

    # log result
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'w') as r:
            r.write('{:s},{:s},{:s},{:s},{:s}'.format('epoch', 'batch', 'loss', 'acc', 'lr'))

    train(train_loader, dev_loader, model, args)



if __name__ == '__main__':
    args = parser.parse_args()
    do_train(args)

