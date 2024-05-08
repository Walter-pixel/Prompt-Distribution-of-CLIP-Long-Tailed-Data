from SwinT_model.swin_transformer import SwinTransformer
# from Resnet_model_dropout.models_224x224.resnet224x224 import *
from Resnet_model_dropout.models_224x224.resnet224x224_DropFcBlocks import *


import numpy as np
from Resnet_model_dropout.models_32x32.resnet_cifar import *
import torch
import torch.nn as nn


'''
Put model to gpus fisrt before to the optimizer
'''


def build_resnet(dataset, loss_type, num_classes, drop_fc_rate, drop_path_rate, drop_block_sz, dropblock_rate,lr1,lr2,
                 weight_decay, momentum, device, mode, ckpt_path=None):
    # Remember, for each drop_path_rate, you need to create a new
    # Bottelneck class (for 224x224 model) or ResNetBasicblock (for 32x32 model)
    # in resnet224x224.py and renset_cifar.py
    # Whilst changing different drop_rate does not need to do so, because drop_rate is only for the last fc layers in Resnet.

    '''

    :param dataset:
    :param num_classes:
    :param drop_fc_rate:
    :param drop_path_rate: it's only implemented on 32x32 resnet for CIFAR100_LT dataset. 224x224 resnet do NOT have this option
    :param drop_block_sz:
    :param dropblock_rate:
    :param lr1:
    :param lr2:
    :param weight_decay:
    :param momentum:
    :param device:
    :param ckpt_path_resnet:
    :return:
    '''

    start_at_epoch = 0

    if dataset =='ImageNet_LT':
        # Imagenet-LT uses Resnet50 train from scratch
        net = resnet50(drop_fc_rate=drop_fc_rate,
                       drop_block_sz=drop_block_sz,
                       dropblock_rate=dropblock_rate,
                       pretrained=False)
        net.fc = nn.Linear(in_features=2048, out_features=num_classes)
        resnet_type = 'resnet50'


    elif dataset == 'iNaturalist18':
        # iNaturalist uses Resnt50 with imagenet pretrained weights
        net = resnet50(drop_fc_rate=drop_fc_rate,
                       drop_block_sz=drop_block_sz,
                       dropblock_rate=dropblock_rate,
                       pretrained=True)
        net.fc = nn.Linear(in_features=2048, out_features=num_classes)
        resnet_type = 'resnet50'

    elif dataset == 'Places_LT':
        # Places-LT uses Resnt152 with coffe imagenet pretrained weights
        net = resnet152(drop_fc_rate=drop_fc_rate,
                        drop_block_sz=drop_block_sz,
                        dropblock_rate=dropblock_rate,
                        pretrained=False)
        # Load caffe model:
        coffe_path = 'resnet152_caffe/resnet152.pth'
        print(f"Loading caffe resnet152 from {coffe_path}")
        weights = torch.load(coffe_path)
        weights = {k: weights[k] if k in weights else net.state_dict()[k]
                   for k in net.state_dict()}
        net.load_state_dict(weights, strict=False)
        net.fc = nn.Linear(in_features=2048, out_features=num_classes)
        resnet_type = 'resnet152'

    elif dataset == 'CIFAR100_LT' or dataset== 'CIFAR10_LT':
        # CIFAR100 uses CIFAR-Resnet32 trained from scratch
        # net = resnet32(drop_fc_rate, drop_block_sz, dropblock_rate, num_classes=num_classes)
        net = resnet32(drop_rate=drop_fc_rate,
                       drop_path_rate=drop_path_rate,
                       num_classes=num_classes)
        resnet_type='resnet32'

    print(f"model is {resnet_type}")



    # set the last layer bias to be b=-log((1-pi)/pi), where pi is the class probability
    if dataset.startswith('CIFAR') and loss_type.startswith('CB_'):
        pi = np.array([1 / num_classes]*num_classes)

        assign = torch.from_numpy(-np.log((1 - pi)/pi)).float()
        net.classifier.bias = nn.Parameter(assign,requires_grad=True)




    '''
    Load resume training checkpoint (if required), and send models to gpus:
    '''
    # # If resume training is required, load saved model states
    # if ckpt_path_resnet != 'None':
    #     print(f"Load checkpoint to model state")
    #     checkpoint = torch.load(ckpt_path_resnet)
    #     net.load_state_dict(checkpoint['backbone_state'])
    #
    # s_params = sum(p.numel() for p in net.parameters())
    # print('|| backbone parameters:%d' % s_params)
    # if torch.cuda.device_count() > 1:
    #     print("\n\nLet's user", torch.cuda.device_count(), "GPUs\n")
    #     net = net.to(device)
    #     net = nn.DataParallel(net)
    # elif torch.cuda.device_count() == 1:
    #     net = net.to(device)
    # net.train()
    # Load resume training checkpoint (if required):

    # '''
    # create optimizer and lr groups
    # '''
    # # lr_grp1, lr_grp2 = set_optimizer_resnet(resnet_type=resnet_type, net=net)
    #
    # params = add_weight_decay(net, l2_value=weight_decay)
    #
    # # optimizer = torch.optim.SGD([
    # #     {"params": lr_grp1.parameters(), "lr": lr1, },
    # #     {"params": lr_grp2.parameters(), "lr": lr2, },
    # # ], weight_decay=weight_decay, momentum=momentum, nesterov=True)
    #
    # optimizer = torch.optim.SGD(params, lr=lr1, momentum=momentum, nesterov=True)
    #
    #
    #
    # # If resuem training is required, load optimizer lr
    # if ckpt_path_resnet != 'None':
    #     print(f"Load checkpoint lr to optimizer")
    #     checkpoint = torch.load(ckpt_path_resnet)
    #     # Must load optimizer state dictionary, otherwise lr will have bump in the begining !
    #     optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
    #     start_at_epoch = checkpoint['epoch'] + 1
    #     print(f'Resume training from epoch {start_at_epoch}')
    #
    # return optimizer, net, start_at_epoch



    if mode == 'load_pretrain':
        '''
        start with pretrained model and user specified lr
        '''
        print(f"Load checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['backbone_state_dict'])

        # Send models to gpus:
        s_params = sum(p.numel() for p in net.parameters())
        print('|| backbone parameters:%d' % s_params)
        if torch.cuda.device_count() > 1:
            print("\n\nLet's user", torch.cuda.device_count(), "GPUs\n")
            net = net.to(device)
            net = nn.DataParallel(net)
        elif torch.cuda.device_count() == 1:
            net = net.to(device)
        net.train()

        # Create optimizer
        params = add_weight_decay(net, l2_value=weight_decay)
        optimizer = torch.optim.SGD(params, lr=lr1, momentum=momentum, nesterov=True)

    elif mode == 'from_scratch':
        '''
        completely train from scratch with use specified lr
        '''
        # Send models to gpus:
        s_params = sum(p.numel() for p in net.parameters())
        print('|| backbone parameters:%d' % s_params)
        if torch.cuda.device_count() > 1:
            print("\n\nLet's user", torch.cuda.device_count(), "GPUs\n")
            net = net.to(device)
            net = nn.DataParallel(net)
            optimizer = torch.optim.SGD(net.module.parameters(), lr=lr1, momentum=momentum, nesterov=True,
                                        weight_decay=weight_decay)
        elif torch.cuda.device_count() == 1:
            net = net.to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=lr1, momentum=momentum, nesterov=True,
                                        weight_decay=weight_decay)
        net.train()
        # Create optimizer
        # params = add_weight_decay(net, l2_value=weight_decay)


    elif mode == 'resume_training':
        '''
        keep training an un-completed result (the epoch specified in the previous script but the training time-out) 
        from the left epoch. In this case, lr is nt frm the user defined lr, but frm the saved checkpoint 
        '''
        print(f"Load checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['backbone_state_dict'])

        # Send models to gpus:
        s_params = sum(p.numel() for p in net.parameters())
        print('|| backbone parameters:%d' % s_params)
        if torch.cuda.device_count() > 1:
            print("\n\nLet's user", torch.cuda.device_count(), "GPUs\n")
            net = net.to(device)
            net = nn.DataParallel(net)
            optimizer = torch.optim.SGD(net.module.parameters(), lr=lr1, momentum=momentum, nesterov=True,
                                        weight_decay=weight_decay)
        elif torch.cuda.device_count() == 1:
            net = net.to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=lr1, momentum=momentum, nesterov=True,
                                        weight_decay=weight_decay)
        net.train()

        # Create optimizer
        # lr_grp1, lr_grp2 = set_optimizer_resnet_bias(net=net)
        # optimizer = torch.optim.SGD([
        #     {"params": lr_grp1.parameters(), "lr": lr1},
        #     {"params": lr_grp2.parameters(), "lr": lr2},
        # ], weight_decay=weight_decay, momentum=momentum, nesterov=True)




        # params = add_weight_decay(net, l2_value=weight_decay)
        # optimizer = torch.optim.SGD(params, lr=lr1, momentum=momentum, nesterov=True)

        # Starts frm save optimizer lr
        if ckpt_path != 'None':
            print(f"Load checkpoint lr to optimizer")
            checkpoint = torch.load(ckpt_path)
            # Must load optimizer state dictionary, otherwise lr will have bump in the begining !
            # optimizer.param_groups[0]['lr'] = checkpoint['optimizer'].param_groups[0]['lr']
            # optimizer.param_groups[1]['lr'] = checkpoint['optimizer'].param_groups[1]['lr']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_at_epoch = checkpoint['epoch'] + 1
            print(f'Resume training from epoch {start_at_epoch}')

    return optimizer, net, start_at_epoch
















def set_optimizer_resnet_bias(net): # net has been in nn.dataparrellel

    '''
    This version can exclude bias from applying weight decay to it,
    The while network uses the same learning rate
    '''
    print(f'\nOptimizer does not apply weight-decay to bias\n')

    if torch.cuda.device_count() > 1:
        bb_syntex = eval('net.module')
    else:
        bb_syntex = eval('net')

    decay_grp = dict()
    no_decay_grp = dict()
    for name, param in net.named_parameters(): # "named_parameters" can detect multi-gpu setting and add "module" in from of parameter name for you
        print('checking {}'.format(name))
        if 'weight' in name:
            decay_grp[name]=param
        else:
            no_decay_grp[name]=param


    #
    # if resnet_type == 'resnet50':
    #     lr_grp1 = nn.ModuleList([])
    #     lr_grp2 = nn.ModuleList([])
    #     lr_grp1.append(bb_syntex.conv1)
    #     lr_grp1.append(bb_syntex.bn1)
    #     lr_grp1.append(bb_syntex.relu)
    #     lr_grp1.append(bb_syntex.maxpool)
    #     lr_grp1.append(bb_syntex.layer1)
    #     lr_grp1.append(bb_syntex.layer2)
    #     lr_grp1.append(bb_syntex.layer3)
    #     lr_grp1.append(bb_syntex.layer4)
    #
    #     lr_grp2.append(bb_syntex.avgpool)
    #     lr_grp2.append(bb_syntex.fc)
    #     try:
    #         lr_grp2.append(bb_syntex.drop)
    #         print(f"Has dropout fc layer")
    #     except:
    #         print(f"No dropout fc layer")
    #     try:
    #         lr_grp2.append(bb_syntex.dropblock)
    #         print(f"Has dropout block layer")
    #     except:
    #         print(f"No drop block layer")
    #
    # elif resnet_type == 'resnet32':
    #     lr_grp1 = nn.ModuleList([])
    #     lr_grp2 = nn.ModuleList([])
    #     lr_grp1.append(bb_syntex.conv_1_3x3)
    #     lr_grp1.append(bb_syntex.bn_1)
    #     lr_grp1.append(bb_syntex.stage_1)
    #     lr_grp1.append(bb_syntex.stage_2)
    #
    #     lr_grp2.append(bb_syntex.stage_3)
    #     lr_grp2.append(bb_syntex.avgpool)
    #     lr_grp2.append(bb_syntex.classifier)
    #     try:
    #         lr_grp2.append(bb_syntex.drop)
    #         print(f"Has dropout fc layer")
    #     except:
    #         print(f"No dropout fc layer")
    #     try:
    #         lr_grp2.append(bb_syntex.dropblock)
    #         print(f"Has dropout block layer")
    #     except:
    #         print(f"No drop block layer")
    #
    # elif resnet_type =='resnet152':
    #     lr_grp1 = nn.ModuleList([])
    #     lr_grp2 = nn.ModuleList([])
    #     lr_grp1.append(bb_syntex.conv1)
    #     lr_grp1.append(bb_syntex.bn1)
    #     lr_grp1.append(bb_syntex.relu)
    #     lr_grp1.append(bb_syntex.maxpool)
    #     lr_grp1.append(bb_syntex.layer1)
    #     lr_grp1.append(bb_syntex.layer2)
    #     lr_grp1.append(bb_syntex.layer3)
    #     lr_grp1.append(bb_syntex.layer4)
    #
    #     lr_grp2.append(bb_syntex.avgpool)
    #     lr_grp2.append(bb_syntex.fc)
    #     try:
    #         lr_grp2.append(bb_syntex.drop)
    #         print(f"Has dropout fc layer")
    #     except:
    #         print(f"No dropout fc layer")
    #     try:
    #         lr_grp2.append(bb_syntex.dropblock)
    #         print(f"Has dropout block layer")
    #     except:
    #         print(f"No drop block layer")

    return decay_grp, no_decay_grp



def set_optimizer_resnet(resnet_type, net, ckpt_path_resnet=None): # net has been in nn.dataparrellel

    if torch.cuda.device_count() > 1:
        bb_syntex = eval('net.module')
    else:
        bb_syntex = eval('net')

    if resnet_type == 'resnet50':
        lr_grp1 = nn.ModuleList([])
        lr_grp2 = nn.ModuleList([])
        lr_grp1.append(bb_syntex.conv1)
        lr_grp1.append(bb_syntex.bn1)
        lr_grp1.append(bb_syntex.relu)
        lr_grp1.append(bb_syntex.maxpool)
        lr_grp1.append(bb_syntex.layer1)
        lr_grp1.append(bb_syntex.layer2)
        lr_grp1.append(bb_syntex.layer3)
        lr_grp1.append(bb_syntex.layer4)

        lr_grp2.append(bb_syntex.avgpool)
        lr_grp2.append(bb_syntex.fc)
        try:
            lr_grp2.append(bb_syntex.drop)
            print(f"Has dropout fc layer")
        except:
            print(f"No dropout fc layer")
        try:
            lr_grp2.append(bb_syntex.dropblock)
            print(f"Has dropout block layer")
        except:
            print(f"No drop block layer")

    elif resnet_type == 'resnet32':
        # while net uses the same lr
        lr_grp1 = nn.ModuleList([])
        lr_grp2 = nn.ModuleList([])
        lr_grp1.append(bb_syntex.conv_1_3x3)
        lr_grp1.append(bb_syntex.bn_1)
        lr_grp1.append(bb_syntex.stage_1)
        lr_grp1.append(bb_syntex.stage_2)
        lr_grp1.append(bb_syntex.stage_3)

        lr_grp1.append(bb_syntex.avgpool)
        lr_grp1.append(bb_syntex.classifier)
        # try:
        #     lr_grp2.append(bb_syntex.drop)
        #     print(f"Has dropout fc layer")
        # except:
        #     print(f"No dropout fc layer")
        # try:
        #     lr_grp2.append(bb_syntex.dropblock)
        #     print(f"Has dropout block layer")
        # except:
        #     print(f"No drop block layer")

    elif resnet_type =='resnet152':
        lr_grp1 = nn.ModuleList([])
        lr_grp2 = nn.ModuleList([])
        lr_grp1.append(bb_syntex.conv1)
        lr_grp1.append(bb_syntex.bn1)
        lr_grp1.append(bb_syntex.relu)
        lr_grp1.append(bb_syntex.maxpool)
        lr_grp1.append(bb_syntex.layer1)
        lr_grp1.append(bb_syntex.layer2)
        lr_grp1.append(bb_syntex.layer3)
        lr_grp1.append(bb_syntex.layer4)

        lr_grp2.append(bb_syntex.avgpool)
        lr_grp2.append(bb_syntex.fc)
        try:
            lr_grp2.append(bb_syntex.drop)
            print(f"Has dropout fc layer")
        except:
            print(f"No dropout fc layer")
        try:
            lr_grp2.append(bb_syntex.dropblock)
            print(f"Has dropout block layer")
        except:
            print(f"No drop block layer")

    return lr_grp1, lr_grp2




def add_weight_decay(net, l2_value, skip_list=()):
    # refer: https://discuss.pytorch.org/t/changing-the-weight-decay-on-bias-using-named-parameters/19132/3

     decay, no_decay = [], []
     for name, param in net.named_parameters():
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
            print(f'decay appends {name}')

     return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]







#=======================================================================================================


# Imagenet-1k pretrained SwinT
def build_swintiny(drop_rate,drop_path_rate,lr1,lr2,weight_decay,momentum, num_classes, device, mode,ckpt_path, dataset):

    ImageNet_pretrained_weights = 'swin_pretrained/swin_tiny_patch4_window7_224.pth'

    EMBED_DIM = 96
    DEPTHS = [2, 2, 6, 2]
    NUM_HEADS = [3, 6, 12, 24]
    WINDOW_SIZE = 7
    net = SwinTransformer(img_size=224,
                          patch_size=4,
                          in_chans=3,
                          num_classes=1000,
                          embed_dim=EMBED_DIM,
                          depths=DEPTHS,
                          num_heads=NUM_HEADS,
                          window_size=WINDOW_SIZE,
                          mlp_ratio=4.,
                          qkv_bias=True,
                          qk_scale=None,
                          drop_rate=drop_rate,
                          drop_path_rate=drop_path_rate,
                          ape=False,
                          patch_norm=True,
                          use_checkpoint=False)

    start_at_epoch = 0

    if dataset == 'ImageNet_LT':
        # Imagenet-LT uses swin-tiny train from scratch
        net.my_remodel(at_layer=4, num_classes=num_classes)


    elif dataset == 'iNaturalist18':
        # iNaturalist uses swin-tiny with imagenet pretrained weights
        checkpoint_imgnet_w = torch.load(ImageNet_pretrained_weights)
        net.load_state_dict(checkpoint_imgnet_w['model'])
        net.my_remodel(at_layer=4, num_classes=num_classes)
        print(f"Load Imagenet pretrained weights from {ImageNet_pretrained_weights}")

    elif dataset == 'Places_LT':
        # Places_LT uses swin-tiny with imagenet pretrained weights
        checkpoint_imgnet_w = torch.load(ImageNet_pretrained_weights)
        net.load_state_dict(checkpoint_imgnet_w['model'])
        net.my_remodel(at_layer=4, num_classes=num_classes)
        print(f"Load Imagenet pretrained weights from {ImageNet_pretrained_weights}")


    elif dataset == 'CIFAR100_LT':
        # CIFAR100 uses swin-tiny train from scratch
        net.my_remodel(at_layer=4, num_classes=num_classes)


    # Load resume training checkpoint (if required):
    if mode=='load_pretrain':
        '''
        start with pretrained model and user specified lr
        '''
        print(f"Load checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['backbone_state'])

        # Send models to gpus:
        s_params = sum(p.numel() for p in net.parameters())
        print('|| backbone parameters:%d' % s_params)
        if torch.cuda.device_count() > 1:
            print("\n\nLet's user", torch.cuda.device_count(), "GPUs\n")
            net = net.to(device)
            net = nn.DataParallel(net)
        elif torch.cuda.device_count() == 1:
            net = net.to(device)
        net.train()

        # Create optimizer
        lr_grp1, lr_grp2 = set_lr_per_layers_swintiny(net=net)
        optimizer = torch.optim.SGD([
            {"params": lr_grp1.parameters(), "lr": lr1},
            {"params": lr_grp2.parameters(), "lr": lr2},
        ], weight_decay=weight_decay, momentum=momentum, nesterov=True)

    elif mode=='from_scratch':
        '''
        completely train from scratch with use specified lr
        '''
        # Send models to gpus:
        s_params = sum(p.numel() for p in net.parameters())
        print('|| backbone parameters:%d' % s_params)
        if torch.cuda.device_count() > 1:
            print("\n\nLet's user", torch.cuda.device_count(), "GPUs\n")
            net = net.to(device)
            net = nn.DataParallel(net)
        elif torch.cuda.device_count() == 1:
            net = net.to(device)
        net.train()
        # Create optimizer
        lr_grp1, lr_grp2 = set_lr_per_layers_swintiny(net=net)
        optimizer = torch.optim.SGD([
            {"params": lr_grp1.parameters(), "lr": lr1},
            {"params": lr_grp2.parameters(), "lr": lr2},
        ], weight_decay=weight_decay, momentum=momentum, nesterov=True)

    elif mode=='resume_training':
        '''
        keep training an un-completed result (the epoch specified in the previous script but the training time-out) 
        from the left epoch. In this case, lr is nt frm the user defined lr, but frm the saved checkpoint 
        '''
        print(f"Load checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['backbone_state'])

        # Send models to gpus:
        s_params = sum(p.numel() for p in net.parameters())
        print('|| backbone parameters:%d' % s_params)
        if torch.cuda.device_count() > 1:
            print("\n\nLet's user", torch.cuda.device_count(), "GPUs\n")
            net = net.to(device)
            net = nn.DataParallel(net)
        elif torch.cuda.device_count() == 1:
            net = net.to(device)
        net.train()

        # Create optimizer
        lr_grp1, lr_grp2 = set_lr_per_layers_swintiny(net=net)
        optimizer = torch.optim.SGD([
            {"params": lr_grp1.parameters(), "lr": lr1},
            {"params": lr_grp2.parameters(), "lr": lr2},
        ], weight_decay=weight_decay, momentum=momentum, nesterov=True)

        # Starts frm save optimizer lr
        if ckpt_path != 'None':
            print(f"Load checkpoint lr to optimizer")
            checkpoint = torch.load(ckpt_path)
            # Must load optimizer state dictionary, otherwise lr will have bump in the begining !
            # optimizer.param_groups[0]['lr'] = checkpoint['optimizer'].param_groups[0]['lr']
            # optimizer.param_groups[1]['lr'] = checkpoint['optimizer'].param_groups[1]['lr']
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            start_at_epoch = checkpoint['epoch'] +1
            print(f'Resume training from epoch {start_at_epoch}')

    return optimizer, net, start_at_epoch



def set_lr_per_layers_swintiny(net):
    '''
    Here we don't use absolute positional embedding (ape) like Swin transformer paper,
    they meniton ape hurts the performance
    '''
    if torch.cuda.device_count() > 1:
        bb_syntex = eval('net.module')
    else:
        bb_syntex = eval('net')

    # let classifier with one lyr upfront to have the same lr
    lr_grp1 = nn.ModuleList([])
    lr_grp2 = nn.ModuleList([])
    lr_grp1.append(bb_syntex.patch_embed)
    lr_grp1.append(bb_syntex.pos_drop)
    lr_grp1.append(bb_syntex.layers[0])
    lr_grp1.append(bb_syntex.layers[1])
    lr_grp1.append(bb_syntex.layers[2])

    lr_grp2.append(bb_syntex.layers[3])
    lr_grp2.append(bb_syntex.norm)
    lr_grp2.append(bb_syntex.avgpool)
    lr_grp2.append(bb_syntex.head)

    return lr_grp1, lr_grp2
