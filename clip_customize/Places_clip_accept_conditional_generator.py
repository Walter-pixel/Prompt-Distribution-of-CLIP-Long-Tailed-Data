import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict


'''
Most of the code borrow from 
https://github.com/KaiyangZhou/CoOp/blob/main/trainers/coop.py
- have modify the way yaml config file is read for initialize our clip model using this script
'''


_tokenizer = _Tokenizer() # use in prompt_learner




def load_clip_to_cpu(vision_backbone_name, download_root: str = None):

    if vision_backbone_name in ["RN50", "RN101","RN50x4","RN50x16", "RN50x64","ViT-B/32","ViT-B/16", "ViT-L/14", "ViT-L/14@336px"] == False:
        print(f"Error: Specified clip vision model is not supported in CLIP ckeckpoint")

    url = clip._MODELS[vision_backbone_name]
    model_path = clip._download(url, download_root or osp.expanduser("./ckpt_clip_github"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model




class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        '''
        tokenized_prompts: each context is a value (token), so dim=[10,77]
        prompts: each context has been mapped to an vector, so dim=[10,77,512]

        '''
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND ,set so because txt transformer's default setting is batch_first=False
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD ,convert back to have batch_dim placed at first.

        # apply layer norm, half dtype friendly
        self.ln_final.to(torch.float32)
        x = self.ln_final(x.to(torch.float32)).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x



#=======================================================================================================================




class Prompt_Generator_Wrapper(nn.Module):
    def __init__(self, cfg, classnames, clip_model, latent_code_dim):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.PROMP_LEARNER.N_CTX #cfg.TRAINER.COOP.N_CTX
        self.ctx_init = cfg.PROMP_LEARNER.CTX_INIT #cfg.TRAINER.COOP.CTX_INIT
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        prompt_prefix = " ".join(["X"] * n_ctx)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.PROMP_LEARNER.CLASS_TOKEN_POSITION

        # Always use class specific context
        cls_token_to_generator = []
        for name in classnames:
            cls_token_to_generator.append(_tokenizer.encode(name))
        # pad 0 at rear
        n = max([len(element) for element in cls_token_to_generator])
        for i in cls_token_to_generator:
            i.extend([0]*(n-len(i)))
        self.register_buffer("conditioned",
                             torch.tensor(cls_token_to_generator, dtype=self.dtype))  # make sure not to update class label tokens during training

        # self.ctx_vectors_empty = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        # self.generator = Generator_conditional(ctx_vectors_empty=self.ctx_vectors_empty,
        #                                        latent_code_dim=latent_code_dim,
        #                                        cls_conditioned=cls_token_to_generator)

        # self.generator = Generator(ctx_vectors_empty=self.ctx_vectors_empty, latent_code_dim=latent_code_dim)

        self.latent_code = nn.Parameter(torch.zeros(self.n_cls, self.n_ctx, latent_code_dim, dtype=self.dtype),
                                        requires_grad=True)  # Each context vector in each class is represented by one latent code.
        # self.latent_code = nn.Parameter(torch.zeros(self.n_classes, latent_code_dim, dtype=self.dtype), requires_grad=True) # each class own its unique latent vector

        nn.init.normal_(self.latent_code, std=0.02)
        n_cls, condi_vec_len = self.conditioned.size()
        self.mn_std_net = nn.Sequential(OrderedDict([
            ('layer0', nn.Linear(latent_code_dim * self.n_ctx + condi_vec_len, 200, dtype=self.dtype)),
            ('act0', nn.LeakyReLU()),
            ('layer1', nn.Linear(200, 200, dtype=self.dtype)),
            ('act1', nn.LeakyReLU()),
            ('layer2', nn.Linear(200, 2 * self.n_ctx * self.ctx_dim, bias=False, dtype=self.dtype))
            # the context matrix to learn has "self.n_ctx * self.len_ctx" elements, and 2 times values since w eneed mean & sig for each element
        ]))
        self.mu_accumulate = None
        self.std_accumulate = None

    def forward(self):
        input = torch.cat((torch.flatten(self.latent_code, start_dim=1), self.conditioned), dim=1)
        input.float()
        input = F.normalize(input, p=2, dim=-1)
        input.type(self.dtype)
        h_e = self.mn_std_net(input)  # will not update self.conditioned as it's registered in buffer
        # Then, we must divide the output to the mean and the log-variance.
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=-1)
        std = torch.exp(0.5 * log_var_e)
        ctx_matrix = mu_e.view(self.n_cls,self.n_ctx,self.ctx_dim)  # Same Gaussian process share fr all classes, dim=[n_ctx, len_ctx]
        log_var_matrix = log_var_e.view(self.n_cls,self.n_ctx,self.ctx_dim)  # Same Gaussian process share fr all classes, dim=[n_ctx, len_ctx]

        # if assume latent distribution is Guassian, and want to regress it to normal Guassian for preventing std blowing up
        # --> than we have close for expression for KL-div in this case
        kl_loss = 0.5 * torch.sum(mu_e ** 2 + std ** 2 - torch.log(1e-8 + std ** 2) - 1) / (torch.numel(mu_e))  # scaler, sum over all classes
        return ctx_matrix, log_var_matrix, kl_loss



    def build_txt_enc_input(self, ctx):
        '''
        ctx: (n_cls, n_ctx, dim) # combine your prompts of each class with the class name --> to feed to text encoder
        '''
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts



#
#
# class Generator_conditional(nn.Module):
#     def __init__(self, ctx_vectors_empty, latent_code_dim, cls_conditioned):
#         super().__init__()
#         print(f"\nprompt generator latent dimension is: {latent_code_dim}\n")
#         self.dtype = ctx_vectors_empty.dtype
#         self.register_buffer("conditioned", cls_conditioned)  # make sure not to update class label tokens during training
#         self.n_classes, self.n_ctx, self.len_ctx = ctx_vectors_empty.size()
#         self.latent_code_matrix = nn.Parameter(torch.zeros(self.n_classes, self.n_ctx, latent_code_dim, dtype=self.dtype), requires_grad=True) # Each context vector in each class is represented by one latent code.
#         # self.latent_code = nn.Parameter(torch.zeros(self.n_classes, latent_code_dim, dtype=self.dtype), requires_grad=True) # each class own its unique latent vector
#
#         nn.init.normal_(self.latent_code, std=0.02)
#
#
#
#
#     # @staticmethod # The reparameterization trick for Gaussians: https://jmtomczak.github.io/blog/4/4_VAE.html
#     # def reparameterization(mu, log_var):
#     #     # The formulat is the following:
#     #     # z = mu + std * epsilon
#     #     # epsilon ~ Normal(0,1)
#     #     # First, we need to get std from log-variance.
#     #     std = torch.exp(0.5 * log_var)
#     #     # Second, we sample epsilon from Normal(0,1).
#     #     eps = torch.randn_like(std)
#     #     # The final output
#     #     return mu + std * eps, mu, std
#     #
#     #     # Sampling procedure.
#
#     def forward(self,):













# class Condi_Mean_STD_Net(nn.Module):
#     def __init__(self,latent_code_dim, cls_conditioned, n_ctx, len_ctx, dtype):
#         super().__init__()
#         self.dtype = dtype
#         self.n_ctx = n_ctx
#         self.len_ctx = len_ctx
#         self.latent_code_dim = latent_code_dim
#         self.cls_conditioned = cls_conditioned
#         n_cls, condi_vec_len = cls_conditioned.size()
#         # Each context vector in each class is represented by one latent code. At the every end, just concat the class token once for this class
#         self.layer0 = nn.Linear(latent_code_dim * self.n_ctx + condi_vec_len, 200, dtype=self.dtype)
#         self.act0 = nn.LeakyReLU()
#         self.layer1 = nn.Linear(200, 200, dtype=self.dtype)
#         self.act1 = nn.LeakyReLU()
#         self.layer2 = nn.Linear(200, 2 * self.n_ctx * self.len_ctx, bias=False, dtype=self.dtype)
#
#     def forward(self, latent_code_matrix, cls_conditioned):
#         input = torch.cat((torch.flatten(latent_code_matrix, start_dim=1),cls_conditioned), dim=1)
#         input.float()
#         input = F.normalize(input, p=2, dim=-1)
#         input.type(self.dtype)
#         x = self.layer0(input)
#         x = self.act0(x)
#         x = self.layer1(x)
#         x = self.act1(x)
#         x = self.layer2(x)
#         return x
