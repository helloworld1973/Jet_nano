import os
from os import listdir
from os.path import isfile, join

mypath = 'E:\\Python Projects\\Jet_nano\\road_following_dataset_xy_stop_2022-08-23_01-52-11\\dataset_xy_stop\\'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for old_name in onlyfiles:
    segments = old_name.split('_')

    #segments[3] = segments[3][:14] + '2' + segments[3][14+1:]
    #new_name = ''
    #for i in range(0, len(segments)-1):
    #    new_name += str(segments[i])+'_'
    new_name = '2_' + segments[3]
    #new_name = segments[len(segments)-1]
    os.rename(mypath+old_name, mypath+new_name)