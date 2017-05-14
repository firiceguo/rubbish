#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import print_function
import os


def dataProcess(infilepath='', outfilepath=''):
    infile = open(infilepath, 'r')
    outfile = open(outfilepath, 'w')

    sample = 0
    cate = [0 for i in range(7)]
    data = []

    for line in infile:
        data.append(line)
        arr = line.split(" ")
        sample += 1
        label = int(float(arr[0]))
        cate[label] += 1
    infile.close()

    print("Sample amount: " + str(sample))
    print("Sample amount of each category: ", cate)

    for dl in data:
        fdic = {}
        arr = dl.split(" ")
        s = str(int(float(arr[0])))
        for ele in arr[1:]:
            pair = ele.split(":")
            fdic[int(pair[0])] = pair[1]
        for i in range(1,97):
            if i in fdic:
                s += " " + str(i) + ":" + fdic[i].split('\n')[0]
                continue
            else:
                fdic[i] = '0.0'
                s += " " + str(i) + ":" + fdic[i]
        outfile.write(s + '\n')


if __name__ == '__main__':
    infilepath = './DS19.libsvm'
    outfilepath = os.getcwd() + '/newData.libsvm'
    dataProcess(infilepath, outfilepath)
