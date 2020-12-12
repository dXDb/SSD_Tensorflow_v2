# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:28:26 2020

@author: CVPR
"""


import os


def makeList(file_dir):
    file_list = os.listdir(file_dir)
    data_list = []

    for file in file_list:
        file = os.path.join(file_dir, file)

        with open(file, 'r') as f:
            data_str = f.readlines()
            for data in data_str:
                data = data.split(' ')
                if not data[0] in data_list:
                    if data[1].split('\n')[0] != "-1":
                        data_list.append(data[0])

    return data_list

def makeTxt(file_dir, data_list):
    file = os.path.join(file_dir, "dataset.txt")
    data_list = sorted(data_list)

    with open(file, 'w') as f:
        for data in data_list:
            f.write(data)
            f.write("\n")

train_dir = './data_list/train'
test_dir  = './data_list/val'

train = makeList(train_dir)
test  = makeList(test_dir)

makeTxt(train_dir, train)
makeTxt(test_dir, test)