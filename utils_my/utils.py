from __future__ import print_function
import torch
import torch.nn.functional as F
import time
import logging
import numpy as np
import os
import math
import torch.optim as optim
import torch.nn as nn

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('|'.join(entries))

        # print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val'+self.fmt+'} ({avg'+self.fmt+'})'
        return fmtstr.format(**self.__dict__)


def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def transform_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h,m,s

def lr_policy(lr_fn):
    def _alr(optimizer, epoch):
        lr = lr_fn(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length):
    def _lr_fn( epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn)


def k_scedular(k_values, k_steps, epoch, current_k):

    zipped = list(zip(k_values, k_steps))

    if len(k_values) == len(k_steps):
        pass
    else:
        print('Error: k_values & k_steps should have same length')

    for i in range(len(zipped)):
        if epoch == zipped[i][1]:
            current_k =zipped[i][0]

    return current_k




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred    = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# def load_checkpoints(net, optimizer, model_path):
#     latestpath = os.path.join(model_path, 'latest.pth.tar')
#     if os.path.exists(latestpath):
#         print('===================>loading the checkpoints from:', latestpath)
#         latest = torch.load(latestpath)
#         last_epoch = latest['epoch']
#         best_epoch = latest['best_epoch']
#         best_top1 = latest['best_top1']
#         best_top5 = latest['best_top5']
#         net.load_state_dict(latest['models_eval'])
#         optimizer.load_state_dict(latest['optim'])
#         return net, optimizer, last_epoch,best_epoch,best_top1,best_top5
#     else:
#         print('====================>Train From Scratch')
#         return net, optimizer, -1, 0, 0, 0
#
# def save_checkpoints(net, optimizer, epoch,best_epoch,best_top1,best_top5, model_path):
#     latest = {}
#     latest['epoch'] = epoch
#     latest['models_eval'] = net.state_dict()
#     latest['optim'] = optimizer.state_dict()
#     latest['best_epoch'] = best_epoch
#     latest['best_top1'] = best_top1
#     latest['best_top5'] = best_top5
#     torch.save(latest, os.path.join(model_path, 'latest.pth.tar'))
#
# def test(test_loader, net):
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
#     net.eval()
#     for idx, (img, target, index) in enumerate(test_loader):
#         img = img.cuda()
#         target = target.cuda()
#         with torch.no_grad():
#             output = net(img)
#         prec1, prec5 = accuracy(output, target, topk=(1,5))
#         top1.update(prec1.item(), img.size(0))
#         top5.update(prec5.item(), img.size(0))
#     return top1.avg, top5.avg
#
# def test2(test_loader, net):
#     top1 = AverageMeter('Acc@1', ':6.2f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
#     net.eval()
#     for idx, (img, target) in enumerate(test_loader):
#         img = img.cuda()
#         target = target.cuda()
#         with torch.no_grad():
#             output = net(img)
#         prec1, prec5 = accuracy(output, target, topk=(1,5))
#         top1.update(prec1.item(), img.size(0))
#         top5.update(prec5.item(), img.size(0))
#     return top1.avg, top5.avg



def load_states_adaptor(net, state_dict, strict=True):
    '''
    This function can load models that saved in
    - multiple gpu modes (contains module wrapper) & single gpu mode

    Refer to: https://blog.csdn.net/weixin_42279044/article/details/85015215

    '''
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]:v for k,v in state_dict.items()}

    if strict:
        net.load_state_dict(state_dict)
    else:
        missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")
    return net



def shot_acc(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20):
    # Train data is only used to  identify many/Medium/Low shot classes
    training_labels = np.array(train_data.dataset.labels).astype(int)

    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] >= many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] <= low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)




def deploy_model_prp_img(imb_clip_model, ckpt_path, lr_gen, lr_im, wd, device, opt_type='adam'):
    # allow multi-gpu and single gpu training
    # this function is for updating both "prompt generator" and "image encoder"
    # use Adam optmizer for all
    # opt_type: sgd, adam

    if torch.cuda.device_count() > 1:
        md = 'imb_clip_model.module'
    else:
        md = 'imb_clip_model'

    print(f"Let's use", torch.cuda.device_count(), "GPUs!")
    if ckpt_path != 'None':
        print(f"Load ckpt to complete unfinished training epoch")
        ckpt = torch.load(ckpt_path)
        imb_clip_model.prompt_learner.load_state_dict(ckpt['prompt_learner_state'])  # load prmpt gen state
        imb_clip_model.image_encoder.load_state_dict(ckpt['im_enc_state'])  # load im_enc state
        if torch.cuda.device_count() > 1:
            imb_clip_model = nn.DataParallel(imb_clip_model)
        imb_clip_model.to(device)

        if opt_type == 'sgd':
            optim_im = optim.SGD(eval(md).image_encoder.parameters(),
                                  lr=lr_im,
                                  weight_decay=wd)
            optim_gen = optim.SGD(eval(md).prompt_learner.mn_std_net.parameters(),
                                  lr=lr_gen,
                                  weight_decay=wd)
            optim_im.load_state_dict(ckpt['optim_im_state_dict'])
            optim_gen.load_state_dict(ckpt['optim_prompt_state_dict'])
        elif opt_type == 'adam':
            optim_im = optim.Adam(eval(md).image_encoder.parameters(),
                                  lr=lr_im, weight_decay=wd)
            optim_gen = optim.Adam(eval(md).prompt_learner.mn_std_net.parameters(),
                                   lr=lr_gen,
                                   weight_decay=wd)  # put "mn_std_net" of the generator into optimizer <will mannually write an update for latent code in the "def train" function later>
            optim_im.load_state_dict(ckpt['optim_im_state_dict'])
            optim_gen.load_state_dict(ckpt['optim_prompt_state_dict'])
        else:
            optim_gen = None
            optim_im = None

        last_epoch = ckpt['at_epoch']
    else:
        print(f"train from epoch 0")
        if torch.cuda.device_count() > 1:
            imb_clip_model = nn.DataParallel(imb_clip_model)
        imb_clip_model.to(device)
        if opt_type == 'sgd':
            optim_im = optim.SGD(eval(md).image_encoder.parameters(),
                                 lr=lr_im,
                                 weight_decay=wd)
            optim_gen = optim.SGD(eval(md).prompt_learner.mn_std_net.parameters(),
                                  lr=lr_gen,
                                  weight_decay=wd)
        elif opt_type == 'adam':
            optim_im = optim.Adam(eval(md).image_encoder.parameters(),
                                  lr=lr_im, weight_decay=wd)
            optim_gen = optim.Adam(eval(md).prompt_learner.mn_std_net.parameters(),
                                   lr=lr_gen,
                                   weight_decay=wd)  # put "mn_std_net" of the generator into optimizer <will mannually write an update for latent code in the "def train" function later>
        else:
            optim_gen = None
            optim_im = None

        last_epoch = -1

    return optim_im, optim_gen, imb_clip_model, last_epoch





def places_deploy_model_prp(imb_clip_model, ckpt_path, lr_gen, wd, device, opt_type='adam'):
    # allow multi-gpu and single gpu training
    # this function is for updating both "prompt generator" and "image encoder"
    # use Adam optmizer for all
    # opt_type: sgd, adam

    if torch.cuda.device_count() > 1:
        md = 'imb_clip_model.module'
    else:
        md = 'imb_clip_model'

    print(f"Let's use", torch.cuda.device_count(), "GPUs!")
    if ckpt_path != 'None':
        print(f"Load ckpt to complete unfinished training epoch")
        ckpt = torch.load(ckpt_path)
        imb_clip_model.prompt_learner.load_state_dict(ckpt['prompt_learner_state'])  # load prmpt gen state
        if torch.cuda.device_count() > 1:
            imb_clip_model = nn.DataParallel(imb_clip_model)
        imb_clip_model.to(device)

        if opt_type == 'sgd':
            optim_gen = optim.SGD(eval(md).prompt_learner.mn_std_net.parameters(),
                                  lr=lr_gen,
                                  weight_decay=wd)
            optim_gen.load_state_dict(ckpt['optim_prompt_state_dict'])
        elif opt_type == 'adam':
            optim_gen = optim.Adam(eval(md).prompt_learner.mn_std_net.parameters(),
                                   lr=lr_gen,
                                   weight_decay=wd)  # put "mn_std_net" of the generator into optimizer <will mannually write an update for latent code in the "def train" function later>
            optim_gen.load_state_dict(ckpt['optim_prompt_state_dict'])
        else:
            optim_gen = None
        last_epoch = ckpt['at_epoch']
    else:
        print(f"train from epoch 0")
        if torch.cuda.device_count() > 1:
            imb_clip_model = nn.DataParallel(imb_clip_model)
        imb_clip_model.to(device)
        if opt_type == 'sgd':
            optim_gen = optim.SGD(eval(md).prompt_learner.mn_std_net.parameters(),
                                  lr=lr_gen,
                                  weight_decay=wd)
        elif opt_type == 'adam':
            optim_gen = optim.Adam(eval(md).prompt_learner.mn_std_net.parameters(),
                                   lr=lr_gen,
                                   weight_decay=wd)  # put "mn_std_net" of the generator into optimizer <will mannually write an update for latent code in the "def train" function later>
        else:
            optim_gen = None

        last_epoch = -1

    return optim_gen, imb_clip_model, last_epoch




























#
#
#
# def deploy_model_prp(imb_clip_model, ckpt_path, lr_gen, wd, device):
#     # allow multi-gpu and single gpu training
#     # this function is for updating ONLY "prompt generator"
#     # use Adam optmizer for all
#     if torch.cuda.device_count() > 1:
#         print(f"Let's use", torch.cuda.device_count(), "GPUs!")
#         if ckpt_path != 'None':
#             print(f"Load ckpt to complete unfinished training epoch")
#             ckpt = torch.load(ckpt_path)
#             imb_clip_model.prompt_learner.load_state_dict(ckpt['prompt_learner_state'])  # load prmpt gen state
#             imb_clip_model = nn.DataParallel(imb_clip_model)
#             imb_clip_model.to(device)
#             optim_gen = optim.Adam(imb_clip_model.module.prompt_learner.generator.mn_std_net.parameters(),
#                                    lr=lr_gen,
#                                    weight_decay=wd)  # put "mn_std_net" of the generator into optimizer <will mannually write an update for latent code in the "def train" function later>
#             optim_gen.load_state_dict(ckpt['optim_prompt_state_dict'])
#             last_epoch = ckpt['at_epoch']
#         else:
#             print(f"train from epoch 0")
#             imb_clip_model = nn.DataParallel(imb_clip_model)
#             imb_clip_model.to(device)
#             optim_gen = optim.Adam(imb_clip_model.module.prompt_learner.generator.mn_std_net.parameters(),
#                                    lr=lr_gen,
#                                    weight_decay=wd)  # put "mn_std_net" of the generator into optimizer <will mannually write an update for latent code in the "def train" function later>
#             last_epoch = -1
#     else:
#         print(f"use one GPU")
#         if ckpt_path != 'None':
#             print(f"Load ckpt to complete unfinished training epoch")
#             ckpt = torch.load(ckpt_path)
#             imb_clip_model.prompt_learner.load_state_dict(ckpt['prompt_learner_state'])  # load prmpt gen state
#             imb_clip_model.to(device)
#             optim_gen = optim.Adam(imb_clip_model.prompt_learner.generator.mn_std_net.parameters(),
#                                    lr=lr_gen, weight_decay=wd)
#             optim_gen.load_state_dict(ckpt['optim_prompt_state_dict'])
#             last_epoch = ckpt['at_epoch']
#         else:
#             print(f"train from epoch 0")
#             imb_clip_model.to(device)
#             optim_gen = optim.Adam(imb_clip_model.prompt_learner.generator.mn_std_net.parameters(),
#                                    lr=lr_gen, weight_decay=wd)
#             last_epoch = -1
#
#     return optim_gen, imb_clip_model, last_epoch