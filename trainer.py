import torch
import numpy as np


import utility
from decimal import Decimal
from tqdm import tqdm
from option import args

from torchvision import transforms 
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import h5py
import os
import cv2
import numpy as np
from torchvision.utils import save_image
from getGazeLoss import *
import torchvision


class Trainer():
    def __init__(self, opt, loader, my_model, my_loss,ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.loss2 = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.dual_models = self.model.dual_models
        self.dual_optimizers = utility.make_dual_optimizer(opt, self.dual_models)
        self.dual_scheduler = utility.make_dual_scheduler(opt, self.dual_optimizers)
        self.error_last = 1e8
        
        # SR test result save
        self.endPoint_flag = True
        # self.endPoint_flag = False

    def train(self):
        # for name, param in self.model.named_parameters():
        #         param.requires_grad = False

        label_txt = open("dataset/integrated_label.txt" , "r")
        labels = label_txt.readlines()
        batch_gaze_loss=[]
        total_detected = 0

        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
       
        self.ckp.set_epoch(epoch)

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        detection_count = 0
        psnr = 0
        for batch, (lr, hr, image_names) in enumerate(self.loader_train):
            # print("loader_train length : ",len(self.loader_train))
            lr, hr = self.prepare(lr, hr)
            
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad() 

            for i in range(len(self.dual_optimizers)):
                self.dual_optimizers[i].zero_grad()

            # forward
            sr = self.model(lr[0])

            sr2lr = []
            for i in range(len(self.dual_models)):
                sr2lr_i = self.dual_models[i](sr[i - len(self.dual_models)])
                sr2lr.append(sr2lr_i)

                
            # ------------ 1. Compute primary loss (Training) -----------
            loss_primary = self.loss(sr[-1], hr)
            for i in range(1, len(sr)):
                loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])
        
            # ------------ 2. Compute dual loss (Training) -----------
            loss_dual = self.loss(sr2lr[0], lr[0])
            for i in range(1, len(self.scale)):
                loss_dual += self.loss(sr2lr[i], lr[i])


            # ------------ 3. Compute PSNR (Training) -----------
            s = np.max(self.scale)
            if isinstance(sr, list): sr_for_psnr = sr[-1]
            sr_for_psnr = utility.quantize(sr_for_psnr, self.opt.rgb_range)

            for i in range(hr.shape[0]):
                        sr_one = torch.reshape(sr_for_psnr[i],(1, sr_for_psnr[i].shape[0],sr_for_psnr[i].shape[1],sr_for_psnr[i].shape[2]))
                        hr_one = torch.reshape(hr[i],(1, hr[i].shape[0],hr[i].shape[1],hr[i].shape[2]))
                        psnr = utility.calc_psnr(
                            sr_one, hr_one, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        psnr +=  psnr/(float(hr.shape[0]) * self.opt.test_every)

            #-----------------clear dir---------------------------
            # clearDir()
            # # # -------------------save SR image ------------
            # image_root_path = "./train_SRImage"
            
            # face_path = os.path.join(image_root_path, "face")
            # os.makedirs(face_path, exist_ok = True)
            # for i in range(len(sr[-1])):
            #     save_image(sr[-1][i]/255, (os.path.join(face_path, image_names[i]+'.jpg')))

            # -------------------save SR image ------------
            # if int(self.scheduler.last_epoch)% 1 == 0: 
            # image_root_path = "./train_SRImage"
            # face_path = os.path.join(image_root_path, "face/%d_epoch/"%self.scheduler.last_epoch)
            # os.makedirs(face_path, exist_ok = True)
            # for i in range(len(sr[-1])):
            #     save_image(sr[-1][i]/255, (os.path.join(face_path, image_names[i]+'.jpg')))
            
            #---------------GenerateEyePatches-------------------
            # SR 결과로부터 양쪽 눈을 찾아 반환한다. 이때 le_cs , re_cs를 tensor가 되어야 한다.
            # 현재 변수 sr은 sr image (<class 'torch.Tensor'>)가 담긴 리스트이다.
            # 이 부분까지는 tensor를 전달한다. -> getGazeLoss.py로 넘어감.

            le_c_list , re_c_list, detected_list = generateEyePatches(sr[-1])
            # le_c_list , re_c_list, detected_list = generateEyePatches(hr)
            # print("Detected images : ",len(detected_list))
            if len(detected_list) == 0:
                continue
            detection_count += len(detected_list)

            gaze_loss, new_image_names  = computeGazeLoss(labels, le_c_list, re_c_list, detected_list, image_names)
            # ------------------compute gaze_loss----------------

            if len(le_c_list[-1]) == 0 or len(re_c_list[-1]) == 0:
                continue
            else:
                left_root_path = "./train_SRImage/left_eye"
                face_path = os.path.join(left_root_path, "face/%d_epoch/"%self.scheduler.last_epoch)
                os.makedirs(face_path, exist_ok = True)
                # print("le_c_list : ",len(le_c_list))
                for i in range(len(le_c_list)):
                    save_image(le_c_list[i]/255, (os.path.join(face_path, new_image_names[i]+'.jpg')))

                right_root_path = "./train_SRImage/right_eye"
                face_path = os.path.join(right_root_path, "face/%d_epoch/"%self.scheduler.last_epoch)
                os.makedirs(face_path, exist_ok = True)
                # print("re_c_list : ",len(re_c_list))
                for i in range(len(re_c_list)):
                    save_image(re_c_list[i]/255, (os.path.join(face_path, new_image_names[i]+'.jpg')))

            # self.ckp.write_log("GE_Loss : "+str(gaze_loss.item()))
            batch_gaze_loss.append(gaze_loss.item())

            # compute total loss
            loss = loss_primary + loss_dual * self.opt.dual_weight + gaze_loss * 0.1
            L1_loss = loss_primary + loss_dual * self.opt.dual_weight
            
            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward(retain_graph=True)
                # print("backward: "+str(loss.item()))
                self.optimizer.step()
                for i in range(len(self.dual_optimizers)):
                    self.dual_optimizers[i].step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
                
            timer_model.hold()

            # print("batch: ", batch)
            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        print("Detected images : ", detection_count)
        total_detected += detection_count
        print("loss_primary : ", loss_primary)
        print("loss_dual : ", loss_dual)
        print("loss_gaze : ", gaze_loss / 8)

          # ------------draw L1 loss graph -----------
        log_path = "experiments/loss_log(training).log"

        if epoch == 1:
            lf =open(log_path, "w+")
        else:
            lf = open(log_path, "a")
        print(L1_loss)
        lf.write(str(L1_loss.item())+ "\n")
        lf.close()

        loss_logs = []
        lf = open(log_path, "r")
        loss_log = lf.readline()
        while(loss_log !=""):
            loss_logs.append(float(loss_log))
            loss_log = lf.readline()
        
        axis = np.linspace(1, epoch, epoch)
        label = 'L1 Loss for Validation'
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, loss_logs, label=label)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('L1 Loss')
        plt.grid(True)
        plt.savefig('./experiments/Training_L1_loss.pdf')
        plt.close(fig)

        # ------------draw psnr graph -----------
        log_path = "experiments/psnr_log(train).log"

        if epoch == 1:
            lf =open(log_path, "w+")
        else:
            lf = open(log_path, "a")
        
        lf.write(str(psnr)+"\n")
        lf.close()

        psnr_logs = []
        lf = open(log_path, "r")
        psnr_log = lf.readline()
        while(psnr_log !=""):
            psnr_logs.append(float(psnr_log))
            psnr_log = lf.readline()
        
        axis = np.linspace(1, epoch, epoch)
        label = 'SR on Training set'
        fig = plt.figure()
        plt.title(label)
        for idx_scale, scale in enumerate([self.opt.scale[0]]):
            plt.plot(
                axis,
                psnr_logs,
                label='Scale {}'.format(scale)
            )
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        # plt.savefig('{}/test_{}.pdf'.format(self.dir, self.opt.data_test))
        plt.savefig('./experiments/Training_PSNR.pdf')
        plt.close(fig)

        # ------------draw gaze loss graph -----------
        epoch_gaze_loss = 0 
        for loss in batch_gaze_loss:
            epoch_gaze_loss += loss
            a = epoch_gaze_loss
        epoch_gaze_loss /= detection_count
        print("epoch_gaze_loss",epoch_gaze_loss)
        print("---------------------------------------------------")
        log_path = "experiments/gaze_loss(train).log"
        
        if epoch == 1:
            lf =open(log_path, "w+")
        else:
            lf = open(log_path, "a")

        lf.write(str(epoch_gaze_loss)+"\n")
        lf.close()

        gaze_logs = []
        lf = open(log_path, "r")
        gaze_log = lf.readline()
        while(gaze_log !=""):
            gaze_logs.append(float(gaze_log))
            gaze_log = lf.readline()
        
        min = gaze_logs[0]
        max = gaze_logs[0]
        for i in range(len(gaze_logs)):
            if gaze_logs[i] <min:
                min = gaze_logs[i]
            if gaze_logs[i] > max:
                max = gaze_logs[i]
        
        if epoch_gaze_loss <= min and epoch_gaze_loss >=max:
            self.endPoint_flag = True

        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        label = 'Gaze on Training set'

        y = gaze_logs
        plt.title('MPII Loss for Training')
        plt.xlabel('epoch')
        plt.ylabel('MPII loss (MSE for angular)')
        plt.grid(True)
        plt.plot(axis, gaze_logs, 'b-')
        plt.savefig('experiments/Training_Gaze_loss.pdf')
        plt.close(fig)

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()
      

    def test(self):
        self.loss2.start_log()
        save_path = "./experiments"
        # Gaze loss 구하기
        label_txt = open("dataset/integrated_label(validation).txt" , "r")
        labels = label_txt.readlines()

        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_test = utility.timer()
        total_gaze = 0
        detected_img = 0
        loss_primary = 0

        # 개별적으로 DRN loss를 on 할 때 사용한다.
        # self.loss.start_log()
        # DRN_loss = 0

        with torch.no_grad():
            scale = max(self.scale)

            for si, s in enumerate([scale]):
                
                f= open('5060_flip o.txt', 'w')
                eval_psnr = 0
                eval_simm =0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    # print(filename)
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    sr = self.model(lr[0])

                    if isinstance(sr, list): sr = sr[-1]

                    sr = utility.quantize(sr, self.opt.rgb_range)

                    if not no_eval:

                        psnr = utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                   
                        hr_numpy = hr[0].cpu().numpy().transpose(1, 2, 0)
                        sr_numpy = sr[0].cpu().numpy().transpose(1, 2, 0)
                        simm = utility.SSIM(hr_numpy, sr_numpy)
                        eval_simm += simm

                        # print('PSNR : ', psnr)
                        
                        eval_psnr +=psnr
                    
                    # ------------ 1. Compute primary loss (Validation) -----------
                    
                    loss_primary += self.loss2(sr, hr)
                    # print("SR의 개수 ",len(sr))
                    # for i in range(1, len(sr)):
                    #     loss_primary += self.loss(sr[i - 1 - len(sr)], lr[i - len(sr)])
                    #     print("각각의 primary loss : ", loss_primary)
                    
                    L1_loss = loss_primary.item() / 4000
                
                    # ------------ 3. Compute gaze loss (Validation) -----------
                    le_c_list , re_c_list, detected_list = generateEyePatches(sr)
        
                    if type(le_c_list) != torch.Tensor :
                        continue

                    # gaze label이 모두 존재하는 test set에 대한 gaze loss를 계산한다.
                    gaze_loss, _, gaze_list = computeGazeLoss_gazetest(labels, le_c_list, re_c_list, detected_list,filename)

                    total_gaze += gaze_loss
                    detected_img += 1

                    # save test results // SR result ---------------------------------
                    
                    if self.endPoint_flag:
                        if self.opt.save_results:
                            self.ckp.save_results_nopostfix(filename, sr, s)

                self.ckp.log[-1, si] = eval_psnr / len(self.loader_test)
                eval_simm = eval_simm / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.2f} (Best: {:.2f} @epoch {})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si],
                        best[0][si],
                        best[1][si] + 1
                    )
                )
                
                # print("TOTAL IMAGES : ", detected_img)
                # print("TOTAL GAZE : ", gaze_loss)
                # print('SIMM:',eval_simm)
                # print('DRN loss : ', DRN_loss)


        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        print("Detected_image : ", detected_img)
        print("loss_primary : ", loss_primary)
        print('Total_gaze : ', total_gaze)
        print('average : ', total_gaze / detected_img)

        # ------------draw gaze loss graph (validation)-----------

        total_gaze /= detected_img
        log_path = "experiments/gaze_loss(validation).log"
        
        epoch = self.scheduler.last_epoch

        if epoch == 1:
            val_log =open(log_path, "w+")
        else:
            val_log = open(log_path, "a")

        val_log.write(str(float(total_gaze))+"\n")
        val_log.close()

        train_gaze_logs = []
        val_log = open(log_path, "r")
        gaze_log = val_log.readline()
        while(gaze_log !=""):
            train_gaze_logs.append(float(gaze_log))
            gaze_log = val_log.readline()
        
        min_ = train_gaze_logs[0]
        max_ = train_gaze_logs[0]
        for i in range(len(train_gaze_logs)):
            if train_gaze_logs[i] < min_:
                min_ = train_gaze_logs[i]
            if train_gaze_logs[i] > max_:
                max_ = train_gaze_logs[i]

        fig = plt.figure()
        axis = np.linspace(1, epoch, epoch)
        plt.clf()
        y = train_gaze_logs
        plt.title('MPII Loss for Validation')
        plt.xlabel('epoch')
        plt.ylabel('MPII loss (MSE for angular)')
        plt.grid(True)
        plt.plot(axis,train_gaze_logs, 'b-')
        plt.savefig('experiments/Validation_Gaze_loss.pdf')
        plt.close(fig)

        # ------------draw L1 loss graph (validation)-----------
        log_path = "experiments/loss_log(validation).log"

        if epoch == 1:
            lf =open(log_path, "w+")
        else:
            lf = open(log_path, "a")
        
        lf.write(str(L1_loss)+ "\n")
        lf.close()

        loss_logs = []
        lf = open(log_path, "r")
        loss_log = lf.readline()
        while(loss_log !=""):
            loss_logs.append(float(loss_log))
            loss_log = lf.readline()
        
        axis = np.linspace(1, epoch, epoch)
        label = 'L1 Loss for Validation'
        fig = plt.figure()
        plt.title(label)
        plt.plot(axis, loss_logs, label=label)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('L1 Loss')
        plt.grid(True)
        plt.savefig('./experiments/Validation_L1_loss.pdf')
        plt.close(fig)

        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()
        for i in range(len(self.dual_scheduler)):
            self.dual_scheduler[i].step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')

        if len(args) > 1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]],

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs