
from multiprocessing.spawn import freeze_support
from rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelVGG

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import utility
import data
import model
import loss  
from option import args
from checkpoint import Checkpoint
from trainer import Trainer


print("main scale >>"+str(args.scale[0]))
utility.set_seed(args.seed)
checkpoint = Checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    gaze_model = GazeEstimationModelVGG(num_out =2).cuda()
    
    # For save Weights of RCAB seperately
    # target = model.model
    # RCAB_num = int(args.n_blocks)

    # route = './DRN_params/(X%d)State_dict.txt'%args.ratio
    # dict_file = open(route,'w')
    # num_list = range(RCAB_num)
    # weight_dic = {}
    # for weight_name in target.state_dict() :
    #     name_space = weight_name.split('.')
    #     if name_space[0] =="up_blocks" and int(name_space[2]) < RCAB_num:
    #         weight_dic[weight_name] = target.state_dict()[weight_name] 
    # dict_file.write(str(weight_dic))
    # dict_file.close()

    # loss = loss.Loss(args, checkpoint) if not args.test_only else None
    loss = loss.Loss(args, checkpoint)
    t = Trainer(args, loader, model, gaze_model, loss, checkpoint)
    def main():
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

    if __name__ == '__main__':  # 중복 방지를 위한 사용
        freeze_support()  # 윈도우에서 파이썬이 자원을 효율적으로 사용하게 만들어준다.
        main()




