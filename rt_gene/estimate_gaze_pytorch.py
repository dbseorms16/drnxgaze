# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

import os

import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

from .estimate_gaze_base import GazeEstimatorBase
from .gaze_estimation_models_pytorch import GazeEstimationModelVGG
from .download_tools import download_gaze_pytorch_models


class GazeEstimator(GazeEstimatorBase):
    def __init__(self, device_id_gaze, model_files):
        super(GazeEstimator, self).__init__(device_id_gaze, model_files)
        download_gaze_pytorch_models()

        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "8"
        tqdm.write("PyTorch using {} threads.".format(os.environ["OMP_NUM_THREADS"]))

        self._transform = transforms.Compose([lambda x: cv2.resize(x, dsize=(224, 224), interpolation=cv2.INTER_CUBIC),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self._models = []

        model_dir_path = "./rt_gene/model_nets"
        model_path = []
        # model_path.append(os.path.join(model_dir_path,"Model_allsubjects1_pytorch.model"))

        model_path.append(os.path.join(model_dir_path, "Alldata_1px_all_epoch=5-val_loss=0.551.model"))
        # for i in range(1,2):
        #     model_path.append(f"../model_nets/Model_allsubjects{i}_pytorch.model")
            # model_path.append(f"../model_nets/self_model{i}.model")

        for ckpt in model_path:
            _model = GazeEstimationModelVGG(num_out=2)
            _model.load_state_dict(torch.load(ckpt))
            
            # for param in _model.features.parameters():
            #     param.requires_grad = False
            
            _model.to(self.device_id_gazeestimation)
            _model.eval()
            self._models.append(_model)

        tqdm.write('Loaded ' + str(len(self._models)) + ' model(s)')

    def estimate_gaze_twoeyes(self, inference_input_left_list, inference_input_right_list, inference_headpose_list):
        transformed_left = torch.stack(inference_input_left_list).to(self.device_id_gazeestimation)
        transformed_right = torch.stack(inference_input_right_list).to(self.device_id_gazeestimation)
        tranformed_head = torch.as_tensor(inference_headpose_list).to(self.device_id_gazeestimation)

        result = [model(transformed_left, transformed_right, tranformed_head).detach().cpu() for model in self._models]
        result = torch.stack(result, dim=1)
        result = torch.mean(result, dim=1).numpy()
        result[:, 1] += self._gaze_offset
        return result

    def input_from_image(self, cv_image):
        return self._transform(cv_image)
