
import os
# os.environ["CUDA_VISIBLE_DEVICES"] ='0,1' #need to be placed before import torch
import torch
from PIL import Image
from torch.backends import cudnn

from data.dataloader import *
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from utils_my.CBLoss import *
from utils_my.utils import *
import argparse
# from Resnet_model.models_224x224.resnet224x224_DropFcBlocks import *
import time
from clip_customize.Places_clip_accept_conditional_generator import*
import yaml
from utils_my.munch import DefaultMunch
import ast
from Resnet_model_my.models_32x32.resnet_cifar import resnet32_w_adpator
from Resnet_model_my.models_224x224.resnet224x224 import resnet152_adaptor, scratch_resnet50_adaptor
import json

'''
README:
- Support muti-gpu & single gpu training
(1) This script only optimize the prompt generator part
(2) The generator is class conditional, input to the generator are (latent code , class token)
(3) this script uses python file: "clip_customize/Places_clip_accept_conditional_generator.py"
(4) the script support train-tst using different number of sample for avg

'''

def main(args):
    #==========================================================================
    if torch.cuda.is_available():
        device="cuda"
        cudnn.benchmark = True
    else: device="cpu"
    # torch.backends.cudnn.benchmark = True
    seed_all(args.seed)
    writer = SummaryWriter(args.tensorboard)
    log_path = os.path.join(args.tensorboard,'clip.log')
    logfile = open(log_path, 'w')
    # flush out the arguments
    argdict = vars(args)
    print(argdict)
    for k, v in argdict.items():
        logfile.write(k + ': ' + str(v) + '\n')
    logfile.write('\n')
    logfile.close()
    #==========================================================================

    data_root = {'ImageNet_LT': '/home/walterl/ml20_scratch/walterl/data/ImageNet2012',
                 'iNaturalist18': '/home/walterl/ml20_scratch/walterl/data/iNaturalist2018',
                 # 'Places_LT': '/home/walterl/ml20_scratch/walterl/data/places365',
                 'Places_LT': '/home/walterl/oy30/walterl/data/Places365',
                 # 'CIFAR100_LT': '/media/wlia0021/hdd/public_dataset/cifar-100-python',
                 # 'CIFAR10_LT': '/media/wlia0021/hdd/public_dataset/cifar-10-python'}
                 'CIFAR100_LT': '/home/walterl/ml20_scratch/walterl/data/cifar100',
                 'CIFAR10_LT': '/home/walterl/ml20_scratch/walterl/data/cifar10'}

    # generated sub-datasets all have test split
    splits = ['train', 'val']
    if args.dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')

    if args.cifar_imb_ratio != 'None':
        cifar_imb_ratio = float(args.cifar_imb_ratio)
    else:
        cifar_imb_ratio = None

    data = {x: load_data(data_root=data_root[args.dataset],
                         dataset=args.dataset,
                         phase=x,
                         batch_size=args.batch_size,
                         batch_size_tst_val=args.batch_size,
                         num_workers=args.workers,
                         top_k_class=None,
                         reverse=False,
                         cifar_imb_ratio=cifar_imb_ratio,
                         resolution=args.resolution,
                         )
            for x in splits}

    loaders = {key: item[1] for key, item in data.items()}
    imb_num_per_cls = data['train'][2]

    # Decide which label belongs to head/mid/low category
    data_groups = {'head_lbs': [], 'mid_lbs': [], 'low_lbs': [], 'head_smpls': 0, 'mid_smpls': 0, 'low_smpls': 0}
    for lb in range(len(imb_num_per_cls)):
        if imb_num_per_cls[lb] >= 100:
            data_groups['head_lbs'].append(lb)
            data_groups['head_smpls'] += imb_num_per_cls[lb]
        elif imb_num_per_cls[lb] <= 20:
            data_groups['low_lbs'].append(lb)
            data_groups['low_smpls'] += imb_num_per_cls[lb]
        else:
            data_groups['mid_lbs'].append(lb)
            data_groups['mid_smpls'] += imb_num_per_cls[lb]


    # read all class label names
    if args.dataset.startswith('CIFAR'):
        classnames = data['train'][1].dataset.classes
    elif args.dataset.startswith('Places'):
        name_txt = open("data/Places_LT_v2/label_name.txt","r").read()
        name_list = name_txt.split("\n")
        name_list = [i.split('/')[2] for i in name_list] # return [class_name id, class_name id,...]
        classnames = [i.split(' ')[0] for i in name_list] # return [class_name, class_name,...]
    elif args.dataset.startswith('ImageNet'):
        with open("data/ImageNet_LT/label_name.txt","r") as f:
            name_dict = ast.literal_eval(f.read())
        name_list = list(name_dict.values())
        classnames = [i.split(',')[0] for i in name_list]
    elif args.dataset.startswith('iNaturalist18'):
        f = open('data/iNaturalist18/categories.json')
        meta_data = json.load(f)
        classnames = [0]*len(meta_data)
        for j in range(len(meta_data)):
            cls_id = int(meta_data[j]['id'])
            classnames[cls_id] = meta_data[j]['kingdom']+' '+meta_data[j]['name']



    clip_model = load_clip_to_cpu(vision_backbone_name='RN50') # load CLIP trained txt & vision encoder
    if args.dtype == 'fp16':
        clip_model.type(torch.float16)
    elif args.dtype == 'fp32':
        clip_model.type(torch.float32)


    with open(args.clip_config_path) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg_dict['PREC'] = args.dtype
        print(cfg_dict)
        # below is from https://github.com/Infinidat/munch
        # due to : https://stackoverflow.com/questions/52570869/load-yaml-as-nested-objects-instead-of-dictionary-in-python
        cfg = DefaultMunch.fromDict(cfg_dict)

    imb_clip_model = CLIP_Lrn_PromptG(cfg,classnames=classnames,clip_model=clip_model) # initiate clip model that uses prompt generator

    # options of image encoder:
    if args.im_enc_type == 'clip_rn50':
        print(f"Image encoder: clip trained resnet-50")
        pass
    elif args.im_enc_type == 'cifar_rn32':
        print(f"Image encoder: cifar resnet-32 from scratch")
        del imb_clip_model.image_encoder
        imb_clip_model.image_encoder = resnet32_w_adpator(adaptor_out_dim=1024).type(clip_model.dtype)

    elif args.im_enc_type == 'caffe_rn152':
        print(f"Image encoder: caffe pretrained resnet-152")
        del imb_clip_model.image_encoder
        imb_clip_model.image_encoder = resnet152_adaptor(caffe_ckpt='Resnet_model_my/resnet152_caffe/resnet152.pth',
                                                  text_enc_dim=1024, dtype=clip_model.dtype)
    elif args.im_enc_type == 'scratch_rn50':
        print(f"Image encoder: scratch resnet-50")
        del imb_clip_model.image_encoder
        imb_clip_model.image_encoder = scratch_resnet50_adaptor(pretrained=False,text_enc_dim=1024).type(clip_model.dtype)# convert image encoder to Halftype as the same text encoder data type in the CLIP

    # prepare model and optimizer, sent to gpus
    optim_gen, imb_clip_model, last_epoch = places_deploy_model_prp(imb_clip_model, ckpt_path=args.ckpt_path,
                                                                           lr_gen=args.lr_gen,
                                                                           wd=args.wd, device=device, opt_type=args.opt_type)

    # create lr-schedular: Must include "last_epoch" keyword for resume training
    if args.lr_type == 'exp':
        lr_sch_gen = torch.optim.lr_scheduler.ExponentialLR(optim_gen, gamma=args.lr_ratio, last_epoch=last_epoch)
    elif args.lr_type == 'multistep':
        lr_sch_gen = torch.optim.lr_scheduler.MultiStepLR(optim_gen, milestones=args.list_steplr, gamma=args.lr_ratio,
                                                          last_epoch=last_epoch)
    elif args.lr_type == 'coslr':
        lr_sch_gen = torch.optim.lr_scheduler.CosineAnnealingLR(optim_gen, T_max=args.epochs, eta_min=0.,
                                                                last_epoch=last_epoch)

    # build the loss
    all_losses = criterion(args, device, imb_num_per_cls=imb_num_per_cls)
    info = info_store(device=device, tf_writer=writer, imb_num_per_cls=imb_num_per_cls, data_groups=data_groups)

    # Training script
    print('\n-----Training Starts-----\n')
    # logfile_mn_std = open(os.path.join(args.tensorboard, 'mean_std.log'), 'w')
    if last_epoch < 0:
        e_start = 0
    else:
        e_start = last_epoch
    for epoch in range(e_start, args.epochs):

        print(f"\n|| Train model epoch {epoch}/{args.epochs} -------------------------------------- ")
        train_start_time = time.time()
        train_multisample(loaders['train'], epoch, args, info, imb_clip_model, optim_gen, all_losses)
        train_epoch_time = time.time() - train_start_time
        print('Train 1epoch time is: {:02}h{:02}m{:02}s'.format(*transform_time(train_epoch_time)))

        lr_sch_gen.step()

        if (epoch % args.infer_at_epoch == 0 and epoch>0) or epoch == args.epochs - 1:
            print(f"\n|| Test model epoch {epoch}/{args.epochs} -------------------------------------- ")
            test_start_time = time.time()
            if args.dataset =="iNaturalist18": # iNaturalist has no testset
                phase = 'val'
            else:
                phase = 'test'
            test_multisample(loaders[phase], loaders['train'], epoch, args, info, imb_clip_model)
            test_epoch_time = time.time() - test_start_time
            print('Test time is {:02}h{:02}m{:02}s'.format(*transform_time(test_epoch_time)))
        info.writer.flush()

    info.writer.close()
    # logfile_mn_std.close()
    print('\n-----KD Training Ends-----\n')






def train_multisample(loader, epoch, args, info, imb_clip_model, optim_gen, all_losses):

    batch_time = AverageMeter('Time', ':.3f')
    record = {'loss_all': AverageMeter('loss_all', ':.3f'),
              'loss_main': AverageMeter('loss_main', ':.3f'),
              'loss_kl': AverageMeter('loss_kl', ':.3f')}
    total_pred = torch.tensor([])
    total_true = torch.tensor([])

    if torch.cuda.device_count() > 1:
        md = 'imb_clip_model.module'
    else:
        md = 'imb_clip_model'

    eval(md).image_encoder.eval()
    eval(md).text_encoder.eval()
    eval(md).prompt_learner.train()

    print(f'Set requires grad false in text & image encoders')
    for param in eval(md).text_encoder.parameters():
        param.requires_grad = False
    for param in eval(md).image_encoder.parameters():
        param.requires_grad = False


    end = time.time()
    for step, (images, labels, _) in enumerate(loader):
        print("\r" + "Epoch: {} Batch :{}".format(epoch, step) + "/" + str(len(loader)),end="", flush=True)
        total_true = torch.cat((total_true, labels), 0)
        images = images.to(info.device)
        labels = labels.to(info.device)
        bz = images.size(0)
        optim_gen.zero_grad()
        logits, kl_loss = imb_clip_model(images,n_smaple=args.n_sample_trn, combine_mode=args.combine_mode)

        # if stochastic tau for logit adj is selected
        if args.loss_type=='logit_adj' and len(args.stoch_tau_range)==1:
            # if pass only 1 argument in range, then it is deterministic tau
            all_losses.tau = args.stoch_tau_range[0]
        elif args.loss_type=='logit_adj' and len(args.stoch_tau_range)==2:
            lower_bound = float(args.stoch_tau_range[0])
            upper_bound = float(args.stoch_tau_range[1])
            all_losses.tau = (upper_bound - lower_bound) * (np.random.rand()) + lower_bound
        else:
            pass

        loss_main = all_losses.mainloss(logits=logits, labels=labels)
        kl_coef = schedular(epoch=epoch, total_e=args.epochs, lr0=args.kl_coef, lr_type=args.lr_type,
                              ratio=args.lr_ratio, list_steplr=args.list_steplr)
        loss_all = loss_main + kl_coef * torch.sum(kl_loss)

        if torch.isnan(loss_main): # for sanity check only
            print(f"labels:\n{labels}")
            print(f"loss:\n{loss_main}")

        _, preds = logits.max(dim=-1)
        total_pred = torch.cat((total_pred,preds.to('cpu')), 0)
        loss_all.backward()
        if args.grad_clip_max != None:
            torch.nn.utils.clip_grad_norm_(eval(md).prompt_learner.parameters(),
                                           max_norm=float(args.grad_clip_max))
        #========================================================================
        '''
        Update latent code
        '''
        lr_latent = schedular(epoch=epoch, total_e=args.epochs, lr0=args.lr_latent, lr_type=args.lr_type,
                              ratio=args.lr_ratio, list_steplr=args.list_steplr)
        # latent_state_before = imb_clip_model.prompt_learner.generator.latent_code.data
        # update latent code of each class in the very front of the prompt generator
        eval(md).prompt_learner.latent_code.data = eval(md).prompt_learner.latent_code.data \
                                                                   - lr_latent * eval(md).prompt_learner.latent_code.grad.data
        # latent_state_after = imb_clip_model.prompt_learner.generator.latent_code.data
        # print(torch.eq(latent_state_before, latent_state_after))


        # ========================================================================
        optim_gen.step()
        record['loss_all'].update(loss_all.detach().item(), labels.size(0))
        record['loss_main'].update(loss_main.detach().item(), labels.size(0))
        record['loss_kl'].update(torch.sum(kl_loss.detach()).item(), labels.size(0))
        batch_time.update(time.time() - end)
        end = time.time()


    # End of this epoch do:--------------------------------------------------
    # Overall accuracy
    if args.dataset.startswith('CIFAR'):
        acc_top1 = (total_pred == total_true).sum().item() / len(total_true)
        print(f"Overall Training Acc: {acc_top1}")
        info.writer.add_scalar('Train/Acc@1', acc_top1, epoch)
    else:
        acc_top1 = (total_pred == total_true).sum().item() / len(total_true)
        print(f"Overall Training Acc: {acc_top1}")
        info.writer.add_scalar('Train/Acc@1', acc_top1, epoch)
        many_acc_top1, \
        median_acc_top1, \
        low_acc_top1 = shot_acc(total_pred, total_true, loader)
        # Top-1 accuracy and additional string
        print_str = [
            'Many_shot_accuracy_top1: %.3f'
            % (many_acc_top1),
            'Median_shot_accuracy_top1: %.3f'
            % (median_acc_top1),
            'Low_shot_accuracy_top1: %.3f'
            % (low_acc_top1)
        ]
        print(print_str)
        info.writer.add_scalar('Train/many_acc_top1', many_acc_top1, epoch)
        info.writer.add_scalar('Train/median_acc_top1', median_acc_top1, epoch)
        info.writer.add_scalar('Train/low_acc_top1', low_acc_top1, epoch)

    info.writer.add_scalar('Train/loss_all', record['loss_all'].avg, epoch)
    info.writer.add_scalar('Train/loss_main', record['loss_main'].avg, epoch)
    info.writer.add_scalar('Train/loss_kl', record['loss_kl'].avg, epoch)
    print(f"\n*- TrainLoss_all: {record['loss_all'].avg}")
    print(f"*- TrainLoss_main: {record['loss_main'].avg}")
    print(f"*- TrainLoss_kl: {record['loss_kl'].avg}")

    info.writer.add_scalar('lr/lr_laten', lr_latent, epoch)
    info.writer.add_scalar('lr/lr_gen', optim_gen.param_groups[0]['lr'], epoch)

    # # add mn_std to histogram: CAUTION: This will not work on multiple GPU ! need to comment out
    # info.writer.add_histogram("Train/Mean", eval(md).prompt_learner.generator.mu_accumulate / bz, bins=200,
    #                           global_step=epoch)
    # info.writer.add_histogram("Train/STD", eval(md).prompt_learner.generator.std_accumulate / bz, bins=200,
    #                           global_step=epoch)


    if epoch % 10 ==0 or epoch >= args.epochs-3:
        Save_Prompt(imb_clip_model, args, epoch,
                    save_path=os.path.join(args.tensorboard, 'prompt_visual_e' + str(epoch) + '.pth')
                    ,optim_prompt=optim_gen, loss=loss_main.detach().item())







def test_multisample(test_loader, train_loader, epoch, args, info, imb_clip_model):
    '''
    This test script sample multiple times in test time
    - n_sample : sample how many times in one forward
    - combine_mode : the way in combining multiple test encoder output,
                     choices = ['avg','max']
    '''
    batch_time = AverageMeter('Time', ':.3f')
    loss_record = AverageMeter('main_loss', ':.3f')
    end = time.time()
    total_pred = torch.tensor([])
    total_true = torch.tensor([])

    imb_clip_model.eval() # set all to be evaluation mode


    if torch.cuda.device_count() > 1:
        md = 'imb_clip_model.module'
    else:
        md = 'imb_clip_model'


    for step, (images, labels, _) in enumerate(test_loader):
        print("\r" + "Epoch: {} Batch :{}".format(epoch, step) + "/" + str(len(test_loader)),
              end="", flush=True)


        total_true = torch.cat((total_true, labels), 0)
        images = images.to(info.device)
        bz = images.size(0)

        with torch.no_grad():
            output, _ = imb_clip_model(images,n_smaple=args.n_sample_tst, combine_mode=args.combine_mode)
            _, preds = output.max(dim=-1)
            total_pred = torch.cat((total_pred,preds.to('cpu')), 0)

        batch_time.update(time.time() - end)
        end = time.time()


    # End of this epoch do:--------------------------------------------------
    # Overall accuracy
    acc_top1 = (total_pred == total_true).sum().item() / len(total_true)
    many_acc_top1, \
    median_acc_top1, \
    low_acc_top1 = shot_acc(total_pred, total_true, train_loader) # train loader is used to distinguish what class is i nthe head/mid/low shot groups

    # Top-1 accuracy and additional string
    print_str = [
        'Many_shot_accuracy_top1: %.3f'
        % (many_acc_top1),
        'Median_shot_accuracy_top1: %.3f'
        % (median_acc_top1),
        'Low_shot_accuracy_top1: %.3f'
        % (low_acc_top1)
    ]
    print(print_str)
    print(f"Overall Testing Acc: {acc_top1}")
    info.writer.add_scalar('Test/Acc@1', acc_top1, epoch)
    info.writer.add_scalar('Test/many_acc_top1', many_acc_top1, epoch)
    info.writer.add_scalar('Test/median_acc_top1', median_acc_top1, epoch)
    info.writer.add_scalar('Test/low_acc_top1', low_acc_top1, epoch)
    # # add mn_std to histogram: CAUTION: This will not work on multiple GPU ! need to comment out
    # info.writer.add_histogram("Test/Mean", eval(md).prompt_learner.generator.mu_accumulate / bz, bins=200,
    #                           global_step=epoch)
    # info.writer.add_histogram("Test/STD", eval(md).prompt_learner.generator.std_accumulate / bz, bins=200,
    #                           global_step=epoch)




class CLIP_Lrn_PromptG(nn.Module):
    '''
    README: this clip implementation intend to "only learn the prompt generator",
    so we set no_grad() to Image Enc to save memory footprint in the forward method
    '''
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.n_classes = len(classnames)
        self.prompt_learner = Prompt_Generator_Wrapper(cfg, classnames, clip_model, latent_code_dim=args.latent_dim) # we assign our prompt generator from here
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual.type(self.dtype)
        self.text_encoder = TextEncoder(clip_model).type(self.dtype)
        self.logit_scale = clip_model.logit_scale

    def forward(self, image, get_feature_only=False, n_smaple=1, combine_mode='None'):
        '''
        n_sample: sample n_sample times
        combine_mode: 'None'--> if you only sample once, then use this mode
                      'avg' --> for class-i, average all loss that incurred by n_sample # of  class-i txt encoder output
                      'max_loss' --> for class-i, pick the largest loss that incurred by n_sample # of  class-i txt encoder output
        '''


        if combine_mode=='avg':
            with torch.no_grad():
                image_features = self.image_encoder(image.type(self.dtype))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            txt_cache = torch.zeros(self.n_classes, image_features.size(dim=-1), dtype=self.dtype).to(image_features.device)  # dim=[n_class, n_sample, emb-len]

            ctx_per_cls, log_var_per_cls, kl_loss = self.prompt_learner()
            tokenized_prompts = self.tokenized_prompts
            for n in range(n_smaple): # modify 12 Oct <ut prompt sampling within the for loop>
                std_per_cls = torch.exp(0.5 * log_var_per_cls)
                eps_per_cls = torch.randn_like(std_per_cls)
                ctx_to_txtenc = ctx_per_cls + eps_per_cls * std_per_cls  # elementwise product
                prompts_to_txtenc = self.prompt_learner.build_txt_enc_input(ctx_to_txtenc)
                text_features = self.text_encoder(prompts_to_txtenc.type(self.dtype),
                                                  tokenized_prompts.type(self.dtype))
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # txt_cache[:, n, :] = text_features # dim=[n_class, n_sample, emb-len]
                txt_cache += text_features  # dim=[n_class, n_sample, emb-len]

            avg_text_features = txt_cache / n_smaple
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ avg_text_features.t()
            return logits, kl_loss #[bz, n_classes], scalar


def schedular(epoch, total_e, lr0, lr_type, ratio=None, list_steplr=None):
    if lr_type == 'multistep':
        if len(list_steplr) == 2:
            if epoch < list_steplr[0]:
                lr = lr0
            elif epoch >= list_steplr[0] and epoch < list_steplr[1]:
                lr = ratio * lr0
            elif epoch >= list_steplr[1]:
                lr = ratio * ratio * lr0
        elif len(list_steplr) == 1:
            if epoch < list_steplr[0]:
                lr = lr0
            else:
                lr = ratio * lr0
        else:
            print(f"Error: number of list_steplr is not supported")

    elif lr_type == 'coslr':
        lr = cosine_decay_with_warmup(global_step=epoch, learning_rate_base=lr0, total_steps=total_e)

    return lr




def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    #refer to: https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b

    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in
        Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
        ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
        a float representing learning rate.
    Raises:
        ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (
            1 + np.cos(
        np.pi * (global_step - warmup_steps - hold_base_rate_steps) /
        float(total_steps - warmup_steps - hold_base_rate_steps)
    )
    )
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return float(np.where(global_step > total_steps, 0.0, learning_rate))



class info_store:
    def __init__(self, device, tf_writer, imb_num_per_cls=None, data_groups={},):
        self.tst_record = {'best_top1': 0, 'best_top5': 0}
        self.val_record = {'best_top1': 0, 'best_top5': 0}
        self.device = device
        self.writer = tf_writer # tensorbaord writer
        self.imb_num_per_cls = imb_num_per_cls
        self.data_groups = data_groups


class criterion():
    def __init__(self, args, device, imb_num_per_cls, data_groups={}):
        self.loss_type = args.loss_type
        self.imb_num_per_cls = imb_num_per_cls # Index starts from class0 to classN
        self.device = device
        self.data_groups = data_groups # Indicates which label belongs to head/mid/low category
        # Below is for weighted CE--------------------------------------------
        self.args = args
        self.instance_w = None
        self.w_category = args.w_category
        # Below is for logit-adj and all its variant versions ----------------
        self.tau = None
        self.base_probs = None

    def mainloss(self, logits, labels):
        # Note, since CBloss was written in the function form, not class, here we define a function wrapper
        if self.loss_type == 'CE':
            return F.cross_entropy(input=logits, target=labels)

        elif self.loss_type == 'wCE':

            # Build the weights first if not initialize it yet
            if self.instance_w == None:
                self.instance_w = self.build_w_for_ce()
                self.instance_w = torch.FloatTensor(self.instance_w).to(self.device)

            # create weighted CE:
            return F.cross_entropy(input=logits, target=labels, weight=self.instance_w)


        elif self.loss_type == 'logit_adj':
            # Initialize
            # self.tau = float(args.logit_adj_tau)
            if self.base_probs == None:
                self.base_probs = torch.from_numpy(self.imb_num_per_cls/sum(self.imb_num_per_cls)
                                          ).to(self.device)

            logits_adjusted = logits + torch.log(self.base_probs.pow(float(self.tau)) + 1e-12)
            return F.cross_entropy(input=logits_adjusted, target=labels)


        elif self.loss_type == 'logit_adj_groups':

            # Initialize
            # self.tau = float(args.logit_adj_tau)
            if self.base_probs == None:
                self.base_probs = np.zeros(len(self.imb_num_per_cls),dtype=float)
                head_prob = len(self.data_groups['head']) / len(self.imb_num_per_cls)
                mid_prob = len(self.data_groups['mid']) / len(self.imb_num_per_cls)
                low_prob = len(self.data_groups['low']) / len(self.imb_num_per_cls)

                for cls in range(len(self.imb_num_per_cls)):
                    if cls in self.data_groups['head']:
                        self.base_probs[cls] = head_prob
                    elif cls in self.data_groups['mid']:
                        self.base_probs[cls] = mid_prob
                    else:
                        self.base_probs[cls] = low_prob

                self.base_probs = torch.from_numpy(self.base_probs).to(self.device)

            logits_adjusted = logits + torch.log(self.base_probs.pow(float(self.tau)) + 1e-12)
            return F.cross_entropy(input=logits_adjusted, target=labels)

        else:
            # CB_focal/CB_softmax/CB_sigmoid
            cb_loss_type = self.loss_type.split("_")[1]
            return CB_loss(labels=labels, logits=logits,
                            samples_per_cls=self.imb_num_per_cls,
                            no_of_classes=self.args.num_classes,
                            loss_type=cb_loss_type,
                            beta=float(self.args.beta),
                            gamma=float(self.args.gamma),
                            device=self.device)

    def build_w_for_ce(self):
        # self.w_category has define which category should assign which weights,
        # based on this, create one-to-one weight/instance mapping for weighted CE loss
        w = [0] * self.args.num_classes
        for lb in range(len(self.imb_num_per_cls)):
            if self.imb_num_per_cls[lb] >= 100:
                w[lb] = float(self.w_category[0])  # assign head-cls weight value from user specified args.w_cateory
            elif self.imb_num_per_cls[lb] <= 20:
                w[lb] = float(self.w_category[2])
            else:
                w[lb] = float(self.w_category[1])
        return w


def Save_Prompt(imb_clip_model, args, epoch, save_path, optim_prompt, loss):
    torch.save({
        'prompt_learner_state': imb_clip_model.module.prompt_learner.state_dict()
                                if torch.cuda.device_count() > 1 else imb_clip_model.prompt_learner.state_dict(),
        'config': args,
        'at_epoch': epoch,
        'optim_prompt_state_dict': optim_prompt.state_dict(),
        'loss': loss,
    }, save_path)










if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PlacesLT')
    parser.add_argument('--seed', type=int, default=88, help='momentum')
    parser.add_argument('--workers', type=int, default=13, help='workers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--infer_at_epoch', type=int, default=10)

    # resume training:
    parser.add_argument('--ckpt_path', type=str, default='None',
                        help='if need to resume training til unfinished eopch, specify the path to saved ckpt .pth here')

    # Dataset ----------------------------------------------------------------------------------------------------------
    parser.add_argument('--dataset', type=str, default='CIFAR10_LT',
                        choices=['iNaturalist18', 'ImageNet_LT', 'Places_LT', 'CIFAR100_LT', 'CIFAR10_LT'])
    parser.add_argument('--cifar_imb_ratio', default='None', help='options are 0.01, 0.02, 0.1, None')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--resolution', type=int, help='image resolution to use')


    # Image encoder structure/txt encoder prompt initialization
    parser.add_argument('--im_enc_type', type=str, choices=['cifar_rn32','clip_rn50','caffe_rn152','scratch_rn50'])
    parser.add_argument('--clip_config_path', type=str, help='path to yaml config for setting up the text encoder prompt')

    # Training parameters
    parser.add_argument('--combine_mode', type=str, choices=['None', 'avg', 'max_within_cls'],
                        help='if the mode None is chosen, then n_sample=1 need to be used')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--T', type=float, help='Temperature paramter')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay 1e-4')
    parser.add_argument('--lr_gen', type=float, default=1e-6, help='backbone lr @ epoch=0')
    parser.add_argument('--lr_latent', type=float, default=1e-6, help='latent code lr lr @ epoch=0')
    parser.add_argument('--lr_type', type=str, default='multistep', help='lr schedular',
                        choices=['exp', 'multistep', 'coslr'])
    parser.add_argument('--lr_ratio', type=float, help='learning rate decay ratio')
    parser.add_argument('--list_steplr', type=int, nargs='+',
                        help='Specify the StepLr changes at what epoch')
    parser.add_argument('--grad_clip_max', default=None)
    parser.add_argument('--kl_coef', type=float, default=1.0,
                        help='coefficient for kl_div loss of prompt generators estimation of mean/std wrt Guassian normal')
    parser.add_argument('--dtype', type=str, help='fata type used', choices=['fp32', 'fp16'], default='fp32')
    parser.add_argument('--latent_dim', type=int, default=3)
    parser.add_argument('--opt_type', type=str, default='adam', choices=['sgd', 'adam'])

    parser.add_argument('--n_sample_trn', type=int, help='sampling how many times during training.')
    parser.add_argument('--n_sample_tst', type=int, help='sampling how many times during testing.')


    # Loss -------------------------------------------------------------------------------------------------------------
    parser.add_argument('--beta', default=None, help='for class balanced loss')
    parser.add_argument('--gamma', default=None, help='for class balanaced loss')
    parser.add_argument('--loss_type', type=str, default=None,
                        choices=['CB_focal', 'CB_sigmoid', 'CB_softmax', 'logit_adj', 'logit_adj_groups', 'CE', 'wCE'],
                        help='for class balanaced loss')
    parser.add_argument('--stoch_tau_range', default="False", nargs='+',
                        help="If false then not using stoch tau, "
                             "If lower and upper bound are given, then sample uniformly between them for tau. "
                             "The list should only contain 2 elements")
    parser.add_argument('--w_category', nargs='+', type=float,
                        help='For weighted CE, manually assign weight to H/M/L groups,'
                             'format: [H_w_value, M_w_value, L_w_value]')

    # Saving ------------------------------------------------------------------------------
    parser.add_argument('--tensorboard', type=str, default='./log/debug')


    args = parser.parse_args()
    main(args)