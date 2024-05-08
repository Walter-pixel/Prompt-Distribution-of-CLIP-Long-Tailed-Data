from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn

from utils.utils import *

from models_224x224_builder.model_builder import *
# from Dino_models.vit.DinoHead_my import *
#
# from data.dataloader import *
# from distiller_zoo import *
#
# from utils.placesLT_utils import *
#
# from utils.placesLT_utils import shot_acc
# from utils.utils import AverageMeter, accuracy
# import os
# from utils.CBLoss import *
# from utils.utils import transform_time
# import time
# from tensorboardX import SummaryWriter
#
# from imblearn.over_sampling import SMOTEN,SMOTE,KMeansSMOTE,BorderlineSMOTE
#
#
#
#
#
# def SMOTE_aug(backbone, train_loader, info, k_neighbors=2, save_path=None):
#
#     # # Run 1epoch to encode complete trainset:
#     # try:
#     #     # Load encoded complete dataset (X,y) if exists
#     #     X = np.load(os.path.join(save_path,'vits8_encode_X.npy'))
#     #     y = np.load(os.path.join(save_path,'vits8_encode_y.npy'))
#     #     print('Existing encoded (X,y) loaded')
#     #     for step, (inputs, labels, _) in enumerate(train_loader):
#     #         att = backbone(inputs, get_att_outputs=True)
#     #         original_h, original_w = att.size()[1], att.size()[2]
#     #         break
#     #
#     # except:
#     for step, (inputs, labels, _) in enumerate(train_loader):
#         print("\r"+"|| Encode training set for SMOTE, @ Batch :{}".format(step)+"/"+str(len(train_loader)),end="",flush=True)
#
#         if step==50: break
#
#         bz = labels.size()[0]
#         backbone.eval()
#         with torch.no_grad():
#             att = backbone(inputs, get_att_outputs=True)
#
#         if step==0:
#             X = att.clone().detach().view(bz, -1).to('cpu')
#             y = labels.clone().detach().to('cpu')
#         else:
#             X = torch.cat((X,att.clone().detach().view(bz,-1).to('cpu')), dim=0)
#             y = torch.cat((y,labels.clone().detach().to('cpu')))
#     original_h, original_w = att.size()[1], att.size()[2]
#
#     X = X.numpy()
#     y = y.numpy()
#     np.save(os.path.join(save_path,'vits8_encode_X.npy'), X)
#     np.save(os.path.join(save_path, 'vits8_encode_y.npy'), y)
#
#     sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
#     X_smo, y_smo = sampler.fit_resample(X, y)
#
#     return X_smo, y_smo, original_h, original_w
#
#
#
#
# class SMOTE_Dataset(Dataset):
#     def __init__(self, X_smo, y_smo, reshape_h=785, reshape_w=384):
#         self.X_smo = X_smo # dim=[#of samples, flattened feature vector]
#         self.y_smo = y_smo
#         self.reshape_h =reshape_h
#         self.reshape_w = reshape_w
#
#
#     def __len__(self):
#         return len(self.self.y_smo)
#
#     def __getitem__(self, index):
#         att_map = self.X_smo[index].view(self.reshape_h, self.reshape_w)
#         label = self.y_smo
#         return att_map, label
#
#
#
