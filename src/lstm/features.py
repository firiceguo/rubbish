#!/usr/bin/python2
# -*- coding: utf-8 -*-


def peakNo(value,minv):
    # find all the peak
    peak = {}
    for i in range(2, 96):
        if (value[i] > value[i-1]) and (value[i] > value[i+1]):
            ld = (value[i] - value[i-1])/(value[i]-minv)
            rd = (value[i] - value[i+1])/(value[i]-minv)
            if ld > 0.3 and rd > 0.3:
                peak[i] = value[i]
    return peak


def valueCate(value, maxv, minv):
    # find the value y in 10 cates  [91, 0, 0, 0, 1, 0, 0, 0, 1, 1]
    vcate = [0 for i in range(10)]
    total = maxv - minv
    if total == 0:
        return [96, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(1, 97):
        cate = int(10*(value[i]-minv)/total)
        if cate == 10:
            cate -= 1
        vcate[cate] += 1
    return vcate


def pknoCell(cell_len, peak):
    no = 96 / cell_len
    pkno = [0 for i in range(no)]
    if len(peak) == 0:
        return pkno
    for k, v in peak.items():
        t = int(k/cell_len)
        pkno[t] += 1
    return pkno


def maxminX(peak):
    index = []

    for k, v in peak.items():
        index.append(k)

    if len(index) <= 1:
        return 0, 0
    index.sort()
    for i in range(1, len(index)):
        index[i] = index[i]-index[i-1]
    return max(index[1:]), min(index[1:])


def maxminY(peak):
    index = []
    for k, v in peak.items():
        index.append(k)
    index.sort()
    if len(index) <= 1:
        return 0, 0
    hd = [0 for j in range(len(index)-1)]
    for i in range(0, len(index)-1):
        hd[i] = peak[index[i+1]]-peak[index[i]]
    return abs(max(hd)), abs(min(hd))


def plateau(value, threshold=0.15, len_x=3):
    """
    :param value: a list 1*96
    :param threshold: record i when value[i] > threshod
    :param len_x: as a plateau when end - start > len_x
    :return num_pla, len_pla, start: #plateau, length of the longest plateau, the begin of the longest plateau
    """
    id = []
    for i in range(len(value)):
        if value[i] > threshold:
            id.append(i)
    if len(id) < len_x:
        return 0, 0, 0
    id = [-10] + id + [-10]
    length = 1
    max_len = 0
    start = 0
    max_start = 0
    num_pla = 0
    for i in range(1, len(id)):
        if id[i] - id[i-1] == 1:
            length += 1
        else:
            if length >= len_x:
                num_pla += 1
                if length > max_len:
                    max_len = length
                    max_start = start
            start = i
            length = 1
    return num_pla, max_len, id[max_start]


def exctFeature(value, cell):
    maxv = max(value[1:])
    minv = min(value[1:])
    peak = peakNo(value, minv)
    new = [maxv, minv, len(peak)]

    vcate = valueCate(value, maxv, minv)
    new.extend(vcate)

    pkno = pknoCell(cell, peak)
    new.extend(pkno)

    maxX, minX = maxminX(peak)
    maxY, minY = maxminY(peak)
    add = [maxX, maxX, maxY, minY]
    new.extend(add)

    th1 = (maxv - minv) * 0.33 + minv
    th2 = (maxv - minv) * 0.67 + minv

    plno, plen, psatrt = plateau(value, th1)
    add = [plno, plen, psatrt]
    new.extend(add)

    plno, plen, psatrt = plateau(value, th2)
    add = [plno, plen, psatrt]
    new.extend(add)
    return new
