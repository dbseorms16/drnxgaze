import os
import math
import torch
import torch.nn as nn
from model.common import DownBlock
import model.drn
from option import args

import sys

def dataparallel(model, gpu_list):
    ngpus = len(gpu_list)
    assert ngpus != 0, "only support gpu mode"
    assert torch.cuda.device_count() >= ngpus, "Invalid Number of GPUs"
    assert isinstance(model, list), "Invalid Type of Dual model"
    for i in range(len(model)):
        if ngpus >= 2:
            model[i] = nn.DataParallel(model[i], gpu_list).cuda()
        else:
            model[i] = model[i].cuda()
    return model


class Model(nn.Module):
    def __init__(self, opt, ckp):
        super(Model, self).__init__()
        print('Making model...')
        self.opt = opt
        self.scale = opt.scale
        self.idx_scale = 0
        self.self_ensemble = opt.self_ensemble
        self.cpu = opt.cpu
        self.device = torch.device('cpu' if opt.cpu else 'cuda')
        self.n_GPUs = opt.n_GPUs

        if self.scale[0] % 2 == 0:
            sf = 2
        else:
            sf = 3

        self.model = drn.make_model(opt).to(self.device)
        self.dual_models = []
        for _ in self.opt.scale:
            dual_model = DownBlock(opt, sf).to(self.device)
            self.dual_models.append(dual_model)
        
        if not opt.cpu and opt.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(opt.n_GPUs))
            self.dual_models = dataparallel(self.dual_models, range(opt.n_GPUs))

        self.load(opt.pre_train, opt.pre_train_dual, cpu=opt.cpu)

        if not opt.test_only:
            print(self.model, file=ckp.log_file)
            print(self.dual_models, file=ckp.log_file)
        
        # compute parameter
        # print(model.)
        ratio = args.ratio
        # num_parameter = self.count_parameters(self.model, ratio)

        # ckp.write_log(f"The number of parameters is {num_parameter / 1000 ** 2:.2f}M")

    def forward(self, x, idx_scale=0):
        self.idx_scale = idx_scale
        target = self.get_model()
        if hasattr(target, 'set_scale'):
            target.set_scale(idx_scale)
        return self.model(x)

    def get_model(self):
        if self.n_GPUs == 1:
            return self.model
        else:
            return self.model.module
    
    def get_dual_model(self, idx):
        if self.n_GPUs == 1:
            return self.dual_models[idx]
        else:
            return self.dual_models[idx].module

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)
    
    # def count_parameters(self, model, ratio):
    #     # DRN's Scale factor
    #     ratio = ratio
    #     # File route of Modified parameters.txt, Scale에 따라서 각각 저장된다.
    #     route = './DRN_params/(X%d)Modified parameters.txt'%ratio
    #     param_file = open(route,'w')
        
    #     # Part 별로 total parameters summation
    #     sub_mean = 0
    #     head = 0
    #     down_block = 0
    #     up_blocks_0 = 0
    #     up_blocks_1 = 0
    #     tail = 0
    #     add_mean = 0
        
    #     for name, p in model.named_parameters():
    #         weight_name = str(name)
    #         param = p.numel()
    #         if 'up_blocks.0' in weight_name:
    #             up_blocks_0 += param  
    #         elif 'up_blocks.1' in weight_name:
    #             up_blocks_1 += param
    #         elif 'tail' in weight_name:
    #             tail += param
    #         elif 'head' in weight_name:
    #             head += param
    #         elif 'down' in weight_name:
    #             down_block += param
    #         elif 'sub_mean' in weight_name:
    #             sub_mean += param
    #         else:
    #             add_mean += param
                
    #         param_file.write(f'name:{name} \n')
    #         param_file.write(f'param.shape:{p.shape} \n')
    #         param_file.write(f'param.shape:{p.numel()} \n')
    #         param_file.write('======================================\n')
        
    #     param_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     param_file.write(f'\nThe number of parameters : {param_sum}')
    #     param_file.write(f' - about {param_sum / 1000 ** 2:.2f}M\n')
    #     param_file.write(f'\nParameters of sub_mean : {sub_mean}\n')
    #     param_file.write(f'\nParameters of head : {head}\n')
    #     param_file.write(f'\nParameters of down_block : {down_block}\n')
    #     param_file.write(f'\nParameters of up_block_0 : {up_blocks_0}')
    #     param_file.write(f' - about {up_blocks_0 / 1000 ** 2:.2f}M\n')
    #     param_file.write(f'\nParameters of up_block_1 : {up_blocks_1}')
    #     param_file.write(f' - about {up_blocks_1 / 1000 ** 2:.2f}M\n')
    #     param_file.write(f'\nParameters of tail : {tail}\n')
    #     param_file.write(f'\nParameters of add_mean : {add_mean}\n')
    #     param_file.write(f'\nFrom sub_mean to add_mean\n')
    #     param_file.write(f'\n{sub_mean} {head} {down_block} {up_blocks_0} {up_blocks_1} {tail} {add_mean}\n')
    #     param_file.close()
    #     return param_sum

            # if self.opt.n_GPUs > 1:
                # return sum(p.numel() for p in model.parameters() if p.requires_grad)
            # param_file.write(f'{name} \n')
            # param_file.write(f'{p.shape} \n')
            # param_file.write(f'{p.numel()} \n')
            # param_file.write('======================================\n')

    def save(self, path, is_best=False):
        target = self.get_model()
        # torch.save: 직렬화된 객체를 디스크에 저장한다.
        # 이 함수는 Python의 pickle 을 사용하여 직렬화한다.
        # 이 함수를 사용하여 모든 종류의 객체의 모델, Tensor 및 사전을 저장할 수 있다.
        torch.save(
            target.state_dict(), 
            os.path.join(path, 'model', args.data_train +'_latest_x'+str(args.scale[len(args.scale)-1])+'.pt')
        )

        RCAB_num = int(args.n_blocks)
        route = './DRN_params/(X%d)State_dict.txt'%args.ratio
        dict_file = open(route,'w')
        num_list = range(RCAB_num)
        weight_dic = {}
        for weight_name in target.state_dict():
            name_space = weight_name.split('.')
            if name_space[0] =="up_blocks" and int(name_space[1]) == 0 and int(name_space[2]) < RCAB_num:
                weight_dic[weight_name] = target.state_dict()[weight_name]
        
        file_name = '(X%d)_up_blocks_weight'%args.ratio
        torch.save(weight_dic, os.path.join(path, 'model', args.data_train + file_name+str(args.scale[len(args.scale)-1])+'.pt'))
        dict_file.write(str(weight_dic))
        dict_file.close()

        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(path, 'model', args.data_train +'_best_x'+str(args.scale[len(args.scale)-1])+'.pt')
            )
        #### save dual models ####
        dual_models = []
        for i in range(len(self.dual_models)):
            dual_models.append(self.get_dual_model(i).state_dict())
        torch.save(
            dual_models,
            os.path.join(path, 'model', args.data_train +'_dual_latest_x'+str(args.scale[len(args.scale)-1])+'.pt')
        )
        if is_best:
            torch.save(
                dual_models,
                os.path.join(path, 'model',args.data_train +'_dual_best_x'+str(args.scale[len(args.scale)-1])+'.pt')
            )

    def load(self, pre_train='.', pre_train_dual='.', cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}
        #### load primal model ####
        if pre_train != '.':
            print('Loading model from {}'.format(pre_train))
            self.get_model().load_state_dict(
                torch.load(pre_train, **kwargs),
                strict=False
            )
        #### load dual model ####
        if pre_train_dual != '.':
            print('Loading dual model from {}'.format(pre_train_dual))
            dual_models = torch.load(pre_train_dual, **kwargs)
            for i in range(len(self.dual_models)):
                self.get_dual_model(i).load_state_dict(
                    dual_models[i], strict=False
                )
