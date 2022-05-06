# -*- ecoding: utf-8 -*-
# @ProjectName: python-classifier-2022-master
# @ModuleName: pcgSegmentation
# @Function: 
# @Author: Eliysiumar
# @Time: 2022/5/3 14:10

import numpy as np
import math
from scipy.signal import kaiserord, lfilter, firwin, butter, find_peaks, hilbert, decimate, spectrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from util.utils import drawPic
import matlab.engine

# 需要1000Hz输入信号

class default_Springer_HSMM_options():
    def __init__(self):
        self.audio_Fs = 1000  # The sampling frequency at which to extract signal features:
        self.audio_segmentation_Fs = 50  # The downsampled frequency. Set to 50 in Springer paper
        self.segmentation_tolerance = 0.1  # Tolerance for S1 and S2 localization, seconds
        self.use_mex = False  # Whether to use the mex code or not: The mex code currently has a bug. This will be fixed asap.
        self.include_wavelet_feature = True  # Whether to use the wavelet function or not:


def running_sum(x):
    """
    Running Sum Algorithm of an input signal is y[n]= x[n] + y[n-1]
    """
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = x[i] + y[i - 1]
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(1, -1)[0]
    return y


def butter_bp_fil(data, lowcut, highcut, fs, order=1):
    """
    Butterworth passband filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


def derivate_1(x):
    """
    Derivate of an input signal as y[n]= x[n] - x[n-1]
    """
    y = np.zeros(len(x))  # Initializate derivate vector
    for i in range(len(x)):
        y[i] = x[i] - x[i - 1]
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(1, -1)[0]
    return y


def do_pcg_seg(pcg_clean, Fs):
    pcg_sum = running_sum(pcg_clean)
    pcg_sum = butter_bp_fil(pcg_sum, 0.01, 2.5, Fs)
    pcg_der = derivate_1(pcg_clean)
    pcg_der = butter_bp_fil(pcg_der, 0.01, 2.5, Fs)

    time_samples = 0.5  # Time to be represented in samples
    mC = int(time_samples * Fs)  # Number of samples to move over the signal
    # re = np.zeros(len(pcg_sum))
    p = find_peaks(pcg_sum, distance=mC)[0]

    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.plot(pcg_clean, label='Signal')
    # plt.plot(pcg_sum, label='Running Sum Algorithm')
    # plt.subplot(2, 1, 2)
    # plt.plot(pcg_clean, label='Signal')
    # plt.plot(pcg_der, label='Signal Sign')
    # plt.legend()
    # plt.show()
    return p


def runSpringerSegmentationAlgorithm(audio_data, Fs, B_matrix=None, pi_vector=None, total_observation_distribution=None):
    # Get PCG Features
    PCG_Features, featuresFs = getSpringerPCGFeatures(audio_data, Fs)


def Homomorphic_Envelope_with_Hilbert(input_signal, sampling_frequency, lpf_frequency=8):
    B_low, A_low = butter(1, 2 * lpf_frequency / sampling_frequency, 'low')
    homomorphic_envelope = np.exp(lfilter(B_low, A_low, np.log(abs(hilbert(input_signal)))))
    homomorphic_envelope[0] = homomorphic_envelope[1]
    return homomorphic_envelope

def normalise_signal(signal):
    mean_of_signal = np.mean(signal)
    standard_deviation = np.std(signal)

    normalised_signal = (signal - mean_of_signal) / standard_deviation
    return normalised_signal

def get_PSD_feature_Springer_HMM(data, sampling_frequency, frequency_limit_low, frequency_limit_high):
    F, T, P = spectrogram(data,fs=sampling_frequency, noverlap=round(sampling_frequency / 80))
    low_limit_position = np.where(abs(F - frequency_limit_low) == np.min(abs(F - frequency_limit_low)))[0][0]
    high_limit_position = np.where(abs(F - frequency_limit_high) == np.min(abs(F - frequency_limit_high)))[0][0]

    # Find the mean PSD over the frequency range of interest:
    psd = np.mean(P[low_limit_position:high_limit_position,:], axis=0)
    return psd

def getSpringerPCGFeatures(audio_data, Fs):
    springer_options = default_Springer_HSMM_options()
    include_wavelet = springer_options.include_wavelet_feature
    featuresFs = springer_options.audio_segmentation_Fs
    # Find the homomorphic envelope
    homomorphic_envelope = Homomorphic_Envelope_with_Hilbert(audio_data, Fs)
    # Downsample the envelope:
    if Fs%featuresFs != 0:
        raise Exception("downSampling_q is not int")
    downSampling_q = int(Fs / featuresFs)
    downsampled_homomorphic_envelope = decimate(homomorphic_envelope, downSampling_q)
    # normalise the envelope
    downsampled_homomorphic_envelope = normalise_signal(downsampled_homomorphic_envelope)

    # Hilbert Envelope
    hilbert_envelope = abs(hilbert(audio_data))
    downsampled_hilbert_envelope = decimate(hilbert_envelope, downSampling_q)
    downsampled_hilbert_envelope = normalise_signal(downsampled_hilbert_envelope)

    # Power spectral density feature
    psd = get_PSD_feature_Springer_HMM(audio_data, Fs, 40, 60)

    PCG_Features = 1
    return  PCG_Features, featuresFs

class pcg_segment():
    def __init__(self):
        self.engine = matlab.engine.start_matlab()
        # self.engine = matlab.engine.connect_matlab()
        # 需在matlab中调用matlab.engine.shareEngine
        self.engine.cd('D:\\UserFolder\\学习资料\\空天院\\课程\\研一下\\医疗电子技术及工程实践 方老师的课\\大作业\\python-classifier-2022-master\\python-classifier-2022-master\\springer_hmm',nargout=0)

    def test_get_pcg_segment(self, audio_data, Fs):
        assigned_states = self.engine.runSpringerSegmentationAlgorithm(audio_data.tolist(), float(Fs))
        assigned_states = np.array(assigned_states).flatten()
        current_state = -1
        start_time = 0
        pcg_segment_list = []
        for state_idx in range(assigned_states.shape[0]):
            _state = int(assigned_states[state_idx])
            if current_state == -1:
                current_state = _state
                start_time = 0
            if current_state != _state:
                if start_time == 0:
                    start_time = state_idx*1.0/Fs
                    current_state = _state
                else:
                    _pcg_seg = np.array([start_time, state_idx*1.0/Fs, current_state])
                    pcg_segment_list.append(_pcg_seg)
                    start_time = state_idx*1.0/Fs
                    current_state = _state
        pcg_segment_list = np.vstack(pcg_segment_list)
        return pcg_segment_list
