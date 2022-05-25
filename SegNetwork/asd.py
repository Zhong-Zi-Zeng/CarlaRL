import os

x_train_path = './data'
x_train_name = os.listdir(x_train_path)
x_train_name.sort(key=lambda x:int(x.split('.')[0]))

i = 0
for file_name in x_train_name:
    if int(file_name[:-4]) != i:
        print(i)
        break
    i += 1

