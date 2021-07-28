import torch
import os 

dir = "./"

path = "./gaze_model_best.pt"

weight = torch.load(path)

new_dict = {}

for k, v in weight.items():
    print(k)
    new_name = str(k)[6:]
    new_dict[new_name] =  v

torch.save(new_dict, os.path.join(dir,'gaze_model_best_remake.pt'))

for k, v in new_dict.items():
    print(k)