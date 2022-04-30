# -*- coding: utf-8 -*-
"""
Author: CHANDAN ACHARYA.
Date : 1 May 2019.
"""
########################### LIBRARIES #########################################
from matplotlib import pyplot as plt
import scipy.io as spio
import scipy.interpolate as interpolate
import numpy as np
import statistics
from scipy.stats import kurtosis
from scipy.stats import skew
from scipy.fft import fft
import scipy.signal as signal
from biosppy.signals import ecg
import neurokit2 as nk
import sys

####################### FEATURE DEFINITIONS ###################################
"""TIME DOMAIN"""


# independent function to calculate RMSSD
# 相邻RR间期均方根
def calc_rmssd(list):
    diff_nni = np.diff(list)  # successive differences
    return np.sqrt(np.mean(diff_nni ** 2))


# 相邻RR间期标准差
def calc_sdsd(list):
    diff_nni = np.diff(list)  # successive differences
    return statistics.stdev(diff_nni)


# independent function to calculate AVRR
# 均值
def calc_avrr(list):
    return sum(list) / len(list)


# independent function to calculate SDRR
# 标准差
def calc_sdrr(list):
    return statistics.stdev(list)


# independent function to calculate SKEW
# 样本偏度
def calc_skew(list):
    return skew(list)


# independent function to calculate KURT
# 峰度
def calc_kurt(list):
    return kurtosis(list)


# nn50
def calc_NNx(list):
    diff_nni = np.diff(list)
    return sum(np.abs(diff_nni) > 50)


# pnn50
def calc_pNNx(list):
    length_int = len(list)
    diff_nni = np.diff(list)
    nni_50 = sum(np.abs(diff_nni) > 50)
    return 100 * nni_50 / length_int


"""NON LINEAR DOMAIN"""


# independent function to calculate SD1
def calc_SD1(list):
    diff_nn_intervals = np.diff(list)
    return np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)


# independent function to calculate SD2
def calc_SD2(list):
    diff_nn_intervals = np.diff(list)
    return np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(
        diff_nn_intervals, ddof=1) ** 2)


# independent function to calculate SD1/SD2
def calc_SD1overSD2(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(
        diff_nn_intervals, ddof=1) ** 2)
    ratio_sd2_sd1 = sd2 / sd1
    return ratio_sd2_sd1


# independent function to calculate CSI
def calc_CSI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(
        diff_nn_intervals, ddof=1) ** 2)
    L = 4 * sd1
    T = 4 * sd2
    return L / T


# independent function to calculate CVI
def calc_CVI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(
        diff_nn_intervals, ddof=1) ** 2)
    L = 4 * sd1
    T = 4 * sd2
    return np.log10(L * T)


# independent function to calculate modified CVI
def calc_modifiedCVI(list):
    diff_nn_intervals = np.diff(list)
    sd1 = np.sqrt(np.std(diff_nn_intervals, ddof=1) ** 2 * 0.5)
    sd2 = np.sqrt(2 * np.std(list, ddof=1) ** 2 - 0.5 * np.std(diff_nn_intervals, ddof=1) ** 2)
    L = 4 * sd1
    T = 4 * sd2
    return L ** 2 / T


# 计算复杂度，这里计算香农熵
def calc_En(list):
    # parameters = nk.complexity_optimize(list, show=False)
    En = nk.entropy_sample(np.array(list))[0]
    return En


"""频域"""


def calc_frequency(y_data, Fs=250, is_interpolate=True, x_data=None):
    y_data = np.array(y_data).flatten()
    if x_data is None:
        x_data = np.arange(0, y_data.shape[0], 1)
    if is_interpolate:
        # 数据插值到对应长度
        interpolation_function = interpolate.interp1d(
            x_data, y_data, kind="quadratic", bounds_error=False, fill_value=([y_data[0]], [y_data[-1]])
        )
        y_new = np.arange(int(np.rint(x_data[-1])))
        y_data = interpolation_function(y_new)

    y_data = y_data - np.mean(y_data)
    N = len(y_data)
    min_frequency = (2 * Fs) / (N / 2)  # for high frequency resolution
    max_frequency = 0.5
    nperseg = int((2 / min_frequency) * Fs)
    if nperseg > N / 2:
        nperseg = int(N / 2)
    frequency, power = signal.welch(
        y_data,
        fs=Fs,
        scaling="density",
        detrend=False,
        nfft=int(nperseg * 2),
        average="mean",
        nperseg=nperseg,
        window="hann"
    )
    power /= np.max(power)
    band_list = {
        'VLF': (0.0033, 0.04),
        'LF': (0.04, 0.15),
        'HF': (0.15, 0.4),
        'VHF': (0.4, 0.5)
    }
    where = (frequency >= min_frequency) & (frequency < max_frequency)
    power = power[where]
    frequency = frequency[where]
    # 总能量TP
    TP = _signal_power_instant_compute(power, frequency, (min_frequency, max_frequency))
    # 极低频能量VLF 0.0033-0.04Hz 对于30秒数据段而言极低频能量点太少，一般需要4分钟
    VLF = _signal_power_instant_compute(power, frequency, band_list['VLF'])
    # 低频能量LF 0.04-0.15Hz
    LF = _signal_power_instant_compute(power, frequency, band_list['LF'])
    # 高频能量HF 0.15-0.4Hz
    HF = _signal_power_instant_compute(power, frequency, band_list['HF'])
    # 极高频能量HF 0.4-0.5Hz
    VHF = _signal_power_instant_compute(power, frequency, band_list['VHF'])
    # LF/HF
    LF_HF = LF / HF
    # LF/TP
    LF_TP = LF / TP
    # HF/TP
    HF_TP = HF / TP
    # VHF/TP
    VHF_TP = VHF/TP
    feature_dict = {
        'VLF': VLF, 'LF': LF, 'HF': HF, 'VHF':VHF,'LF/HF': LF_HF, 'LF/TP': LF_TP, 'HF/TP': HF_TP, 'VHF/TP':VHF_TP
    }
    return feature_dict


def _signal_power_instant_compute(power, frequency, band):
    """Also used in other instances"""
    where = (frequency >= band[0]) & (frequency < band[1])
    power = np.trapz(y=power[where], x=frequency[where])
    return np.nan if power == 0.0 else power


# sliding window function
def slidingWindow(sequence, winSize, step):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""

    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence\
                        length.")
    # Pre-compute number of chunks to emit
    numOfChunks = ((len(sequence) - winSize) / step) + 1
    # Do the work
    for i in range(0, int(numOfChunks) * step, step):
        yield sequence[i:i + winSize]


####################### FEATURE EXTRACTION ####################################

def feature_extract(peaks=None, list_rri=None, feature_name_list=None, Fs=250):
    feature_dict = {}
    # if feature_name in ["RMSSD", "SDSD", "AVRR", "SDRR", "SKEW", "KURT", "NNx", "pNNx", "SD1", "SD2",
    #                     "SD1/SD2", "CSI", "CVI", "modifiedCVI"]:
    # 线性时域特征
    if "RMSSD" in feature_name_list:
        feature_dict["RMSSD"] = calc_rmssd(list_rri)
        # feature_name_list.remove("RMSSD")
    if "SDSD" in feature_name_list:
        feature_dict["SDSD"] = calc_sdsd(list_rri)
        # feature_name_list.remove("SDSD")
    if "AVRR" in feature_name_list:
        feature_dict["AVRR"] = calc_avrr(list_rri)
        # feature_name_list.remove("AVRR")
    if "SDRR" in feature_name_list:
        feature_dict["SDRR"] = calc_sdrr(list_rri)
        # feature_name_list.remove("SDRR")
    if "SKEW" in feature_name_list:
        feature_dict["SKEW"] = calc_skew(list_rri)
        # feature_name_list.remove("SKEW")
    if "KURT" in feature_name_list:
        feature_dict["KURT"] = calc_kurt(list_rri)
        # feature_name_list.remove("KURT")
    if "NNx" in feature_name_list:
        feature_dict["NNx"] = calc_NNx(list_rri)
        # feature_name_list.remove("NNx")
    if "pNNx" in feature_name_list:
        feature_dict["pNNx"] = calc_pNNx(list_rri)
        # feature_name_list.remove("pNNx")
    # 非线性时域特征
    if "SD1" in feature_name_list:
        feature_dict["SD1"] = calc_SD1(list_rri)
        # feature_name_list.remove("SD1")
    if "SD2" in feature_name_list:
        feature_dict["SD2"] = calc_SD2(list_rri)
        # feature_name_list.remove("SD2")
    if "SD1/SD2" in feature_name_list:
        feature_dict["SD1/SD2"] = calc_SD1overSD2(list_rri)
        # feature_name_list.remove("SD1/SD2")
    if "CSI" in feature_name_list:
        feature_dict["CSI"] = calc_CSI(list_rri)
        # feature_name_list.remove("CSI")
    if "CVI" in feature_name_list:
        feature_dict["CVI"] = calc_CVI(list_rri)
        # feature_name_list.remove("CVI")
    if "modifiedCVI" in feature_name_list:
        feature_dict["modifiedCVI"] = calc_modifiedCVI(list_rri)
        # feature_name_list.remove("modifiedCVI")
    if "En" in feature_name_list:
        feature_dict["En"] = calc_En(list_rri)
    # 频域特征
    if len(set(feature_name_list) & {'VLF', 'LF', 'HF', 'LF/HF', 'LF/TP', 'HF/TP'}) > 0:
        # frequency_dict = calc_frequency(peaks, Fs)
        frequency_dict = nk.hrv_frequency(peaks=peaks, sampling_rate=Fs)

        if 'VLF' in feature_name_list:
            feature_dict['VLF'] = float(frequency_dict.get('HRV_VLF'))
            # feature_name_list.remove('VLF')
        if 'LF' in feature_name_list:
            feature_dict['LF'] = float(frequency_dict.get('HRV_LF'))
            # feature_name_list.remove('LF')
        if 'HF' in feature_name_list:
            feature_dict['HF'] = float(frequency_dict.get('HRV_HF'))
            # feature_name_list.remove('HF')
        if 'LF/HF' in feature_name_list:
            feature_dict['LF/HF'] = float(frequency_dict.get('HRV_LFHF'))
            # feature_name_list.remove('LF/HF')
        if 'LF/TP' in feature_name_list:
            feature_dict['LF/TP'] = float(frequency_dict.get('HRV_LFn'))
            # feature_name_list.remove('LF/TP')
        if 'HF/TP' in feature_name_list:
            feature_dict['HF/TP'] = float(frequency_dict.get('HRV_HFn'))
            # feature_name_list.remove('HF/TP')

        # if len(set(feature_name_list) & {'VLF', 'LF', 'HF', 'LF/HF', 'LF/TP', 'HF/TP'}) > 0:
        #     frequency_dict = calc_frequency(y_data=list_rri, Fs=Fs,
        #                                     x_data=peaks[1:]
        #                                     )
        #
        #     feature_dict['VLF_1'] = float(frequency_dict.get('VLF'))
        #     feature_dict['LF_1'] = float(frequency_dict.get('LF'))
        #     feature_dict['HF_1'] = float(frequency_dict.get('HF'))
        #     feature_dict['LF/HF_1'] = float(frequency_dict.get('LF/HF'))
        #     feature_dict['LF/TP_1'] = float(frequency_dict.get('LF/TP'))
        #     feature_dict['HF/TP_1'] = float(frequency_dict.get('HF/TP'))
    return feature_dict


def feature_extract_rsp(rsp_rate, info, feature_name_list=None, Fs=100):
    feature_dict = {}
    frequency_dict = nk.rsp_rrv(rsp_rate, info, sampling_rate=Fs)
    if 'VLF' in feature_name_list:
        feature_dict['VLF'] = float(frequency_dict.get('RRV_VLF'))
        # feature_name_list.remove('VLF')
    if 'LF' in feature_name_list:
        feature_dict['LF'] = float(frequency_dict.get('RRV_LF'))
        # feature_name_list.remove('LF')
    if 'HF' in feature_name_list:
        feature_dict['HF'] = float(frequency_dict.get('RRV_HF'))
        # feature_name_list.remove('HF')
    if 'LF/HF' in feature_name_list:
        feature_dict['LF/HF'] = float(frequency_dict.get('RRV_LFHF'))
        # feature_name_list.remove('LF/HF')
    if 'LF/TP' in feature_name_list:
        feature_dict['LF/TP'] = float(frequency_dict.get('RRV_LFn'))
        # feature_name_list.remove('LF/TP')
    if 'HF/TP' in feature_name_list:
        feature_dict['HF/TP'] = float(frequency_dict.get('RRV_HFn'))
    return feature_dict


########################### PLOTTING ##########################################
def plot_features(featureList, label):
    plt.title(label)
    plt.plot(featureList)
    plt.show()


###################### CALLING FEATURE METHODS ################################
def browsethroughSeizures(list_rri, winSize, step):
    features = ["RMSSD", "SDSD", "AVRR", "SDRR", "SKEW", "KURT", "NNx", "pNNx", "SD1", "SD2", \
                "SD1/SD2", "CSI", "CVI", "modifiedCVI"]
    for item in features:
        featureList = feature_extract(list_rri, item)
        plot_features(featureList, item)

#################### BAYESIAN CHANGE POINT DETECTION ##########################
####inspired by https://github.com/hildensia/bayesian_changepoint_detection
# def bayesianOnFeatures(list_rri, winSize, step):
#     features = ["RMSSD","SDSD", "AVRR", "SDRR", "SKEW", "KURT", "NNx", "pNNx", "SD1", "SD2",
#                 "SD1/SD2", "CSI", "CVI", "modifiedCVI"]
#     for item in features:
#         featureList = feature_extract(list_rri, item)
#         featureList = np.asanyarray(featureList)
#         Q, P, Pcp = ocpd.offline_changepoint_detection \
#             (featureList, partial(ocpd.const_prior, l=(len(featureList) + 1)) \
#              , ocpd.gaussian_obs_log_likelihood, truncate=-40)
#         fig, ax = plt.subplots(figsize=[15, 7])
#         ax = fig.add_subplot(2, 1, 1)
#         ax.set_title(item)
#         ax.plot(featureList[:])
#         ax = fig.add_subplot(2, 1, 2, sharex=ax)
#         ax.plot(np.exp(Pcp).sum(0))

#################### CHANGE POINT DETECTION ##########################
