from glob import glob
import shutil

label_txt = open("C:/Users/admin/Desktop/DRNXGAZE/dataset/integrated_label(validation).txt" , "r")
labels = label_txt.readlines()
a = glob("C:/Users/admin/Desktop/DRNXGAZE/dataset/benchmark/face_test/LR_bicubic/x2/*.jpg")

file1 = []

for aa in a :
    b = aa.split('\\')[1]
    bb = b.split('.')[0]
    file1.append(bb)
    print(bb)

print(len(labels))
for name in labels:
    filename = name.split(',')[0]
    filename = filename+'x2'
    if filename in file1:
        shutil.copyfile("C:/Users/admin/Desktop/DRNXGAZE/dataset/benchmark/face_test/LR_bicubic/x2/"+filename+".jpg",
                         "C:/Users/admin/Desktop/DRNXGAZE/dataset/benchmark/face_test/LR_bicubic/x22/"+filename+".jpg")



