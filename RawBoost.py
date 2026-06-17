#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RawBoost: A Raw Data Boosting and Augmentation Method
用于自动说话人验证反欺诈的原始数据增强方法

本文件负责：
1. RawBoost 官方风格参数 RawBoostArgs
2. RawBoost 三类增强：
   - LnL: Linear and non-linear convolutive noise
   - ISD: Impulsive signal-dependent additive noise
   - SSI: Stationary signal-independent additive noise
3. process_Rawboost_feature 统一增强入口
"""

import copy
import numpy as np
from scipy import signal


class RawBoostArgs:
    """
    Official-style RawBoost data augmentation configuration.

    官方 RawBoost 代码中这些参数原本通过 argparse 传入。
    这里集中放到 RawBoost.py，避免 main.py 里堆很多增强参数。
    """

    def __init__(self, algo=4):
        self.algo = algo

        # LnL / SSI shared filter parameters
        self.N_f = 5
        self.nBands = 5
        self.minF = 20
        self.maxF = 8000
        self.minBW = 100
        self.maxBW = 1000
        self.minCoeff = 10
        self.maxCoeff = 100

        # Gain settings for LnL
        self.minG = 0
        self.maxG = 0
        self.minBiasLinNonLin = 5
        self.maxBiasLinNonLin = 20

        # ISD
        self.P = 10
        self.g_sd = 2

        # SSI
        self.SNRmin = 10
        self.SNRmax = 40


def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        return int(y.item())
    return float(y.item())


def normWav(x, always):
    x = np.asarray(x, dtype=np.float32)
    max_abs = np.amax(np.abs(x))

    if max_abs < 1e-12:
        return x

    if always:
        x = x / max_abs
    elif max_abs > 1:
        x = x / max_abs

    return x.astype(np.float32)


def genNotchCoeffs(nBands, minF, maxF, minBW, maxBW,
                   minCoeff, maxCoeff, minG, maxG, fs):
    b = 1

    for _ in range(0, nBands):
        fc = randRange(minF, maxF, 0)
        bw = randRange(minBW, maxBW, 0)
        c = randRange(minCoeff, maxCoeff, 1)

        if c / 2 == int(c / 2):
            c = c + 1

        f1 = fc - bw / 2
        f2 = fc + bw / 2

        if f1 <= 0:
            f1 = 1 / 1000

        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000

        b = np.convolve(
            signal.firwin(
                c,
                [float(f1), float(f2)],
                window='hamming',
                fs=fs
            ),
            b
        )

        G = randRange(minG, maxG, 0)
        _, h = signal.freqz(b, 1, fs=fs)
        max_h = np.amax(np.abs(h))
        if max_h < 1e-12:
            max_h = 1e-12

        b = pow(10, G / 20) * b / max_h

    return np.asarray(b, dtype=np.float32)


def filterFIR(x, b):
    """
    官方 RawBoost 风格：
    zero-padding + FIR filtering + center cropping。

    不建议直接 return signal.lfilter(b, 1, x)，否则会保留额外滤波延迟。
    """
    x = np.asarray(x, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2): int(y.shape[0] - N / 2)]

    return y.astype(np.float32)


def LnL_convolutive_noise(x, N_f, nBands, minF, maxF, minBW, maxBW,
                          minCoeff, maxCoeff, minG, maxG,
                          minBiasLinNonLin, maxBiasLinNonLin, fs):
    """
    Linear and non-linear convolutive noise.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.zeros(x.shape[0], dtype=np.float32)

    cur_minG = minG
    cur_maxG = maxG

    for i in range(0, N_f):
        if i == 1:
            cur_minG = cur_minG - minBiasLinNonLin
            cur_maxG = cur_maxG - maxBiasLinNonLin

        b = genNotchCoeffs(
            nBands, minF, maxF, minBW, maxBW,
            minCoeff, maxCoeff, cur_minG, cur_maxG, fs
        )

        y += filterFIR(np.power(x, (i + 1)), b)

    y = y - np.mean(y)
    y = normWav(y, 0)

    return y.astype(np.float32)


def ISD_additive_noise(x, P, g_sd):
    """
    Impulsive signal-dependent additive noise.
    """
    x = np.asarray(x, dtype=np.float32)

    beta = randRange(0, P, 0)
    y = copy.deepcopy(x)

    x_len = x.shape[0]
    n = int(x_len * (beta / 100))

    if n <= 0:
        return y.astype(np.float32)

    p = np.random.permutation(x_len)[:n]

    t1 = 2 * np.random.rand(p.shape[0]) - 1
    t2 = 2 * np.random.rand(p.shape[0]) - 1
    f_r = np.multiply(t1, t2)

    r = g_sd * x[p] * f_r
    y[p] = x[p] + r

    y = normWav(y, 0)

    return y.astype(np.float32)


def SSI_additive_noise(x, SNRmin, SNRmax, nBands, minF, maxF, minBW, maxBW,
                       minCoeff, maxCoeff, minG, maxG, fs):
    """
    Stationary signal-independent additive noise.
    """
    x = np.asarray(x, dtype=np.float32)

    noise = np.random.normal(0, 1, x.shape[0]).astype(np.float32)

    b = genNotchCoeffs(
        nBands, minF, maxF, minBW, maxBW,
        minCoeff, maxCoeff, minG, maxG, fs
    )

    noise = filterFIR(noise, b)
    noise = normWav(noise, 1)

    SNR = randRange(SNRmin, SNRmax, 0)

    noise_norm = np.linalg.norm(noise, 2)
    x_norm = np.linalg.norm(x, 2)

    if noise_norm < 1e-12 or x_norm < 1e-12:
        return x.astype(np.float32)

    noise = noise / noise_norm * x_norm / 10.0 ** (0.05 * SNR)
    y = x + noise

    return y.astype(np.float32)


def process_Rawboost_feature(feature, sr, args, algo=None):
    """
    应用 RawBoost 数据增强。

    algo:
        0: no augmentation
        1: LnL
        2: ISD
        3: SSI
        4: LnL + ISD + SSI
        5: LnL + ISD
        6: LnL + SSI
        7: ISD + SSI
        8: LnL || ISD
    """
    feature = np.asarray(feature, dtype=np.float32)

    if algo is None:
        algo = int(getattr(args, "algo", 0))
    else:
        algo = int(algo)

    if algo == 1:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin,
            sr
        )

    elif algo == 2:
        feature = ISD_additive_noise(
            feature,
            args.P,
            args.g_sd
        )

    elif algo == 3:
        feature = SSI_additive_noise(
            feature,
            args.SNRmin, args.SNRmax,
            args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff,
            args.minG, args.maxG,
            sr
        )

    elif algo == 4:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin,
            sr
        )

        feature = ISD_additive_noise(
            feature,
            args.P,
            args.g_sd
        )

        feature = SSI_additive_noise(
            feature,
            args.SNRmin, args.SNRmax,
            args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff,
            args.minG, args.maxG,
            sr
        )

    elif algo == 5:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin,
            sr
        )

        feature = ISD_additive_noise(
            feature,
            args.P,
            args.g_sd
        )

    elif algo == 6:
        feature = LnL_convolutive_noise(
            feature,
            args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin,
            sr
        )

        feature = SSI_additive_noise(
            feature,
            args.SNRmin, args.SNRmax,
            args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff,
            args.minG, args.maxG,
            sr
        )

    elif algo == 7:
        feature = ISD_additive_noise(
            feature,
            args.P,
            args.g_sd
        )

        feature = SSI_additive_noise(
            feature,
            args.SNRmin, args.SNRmax,
            args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW,
            args.minCoeff, args.maxCoeff,
            args.minG, args.maxG,
            sr
        )

    elif algo == 8:
        feature1 = LnL_convolutive_noise(
            feature,
            args.N_f, args.nBands, args.minF, args.maxF,
            args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
            args.minG, args.maxG,
            args.minBiasLinNonLin, args.maxBiasLinNonLin,
            sr
        )

        feature2 = ISD_additive_noise(
            feature,
            args.P,
            args.g_sd
        )

        feature = normWav(feature1 + feature2, 0)

    else:
        feature = feature

    feature = np.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=-1.0)

    return feature.astype(np.float32)