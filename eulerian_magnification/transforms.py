import numpy as np
import scipy

def rgb2ntsc(img):
    # RGB to YIQ transformation
    X = img.reshape(-1,3)
    A = np.array([[0.299,0.587,0.114],[0.596,-.274,-.322],[0.211,-.523,0.312]])
    ntsc = np.dot(A,X.T).T
    return ntsc.reshape(img.shape)

def ntsc2rgb(ntsc):
    # YIQ to RGB transformation
    X = ntsc.reshape(-1,3)
    A = np.linalg.inv(np.array([[0.299,0.587,0.114],[0.596,-.274,-.322],[0.211,-.523,0.312]]))
    img = np.dot(A,X.T).T
    return img.reshape(ntsc.shape)

def uint8_to_float(img):
    result = np.ndarray(shape=img.shape, dtype='float')
    result[:] = img * (1. / 255)
    return result

def float_to_uint8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = np.clip(img * 255,0,255)
    return result


def float_to_int8(img):
    result = np.ndarray(shape=img.shape, dtype='uint8')
    result[:] = np.clip((img * 255) - 127,0,255)
    return result

def temporal_bandpass_filter(data, fps, freq_min=0.833, freq_max=1, axis=0, amplification_factor=1):
    print("Applying bandpass between " + str(freq_min) + " and " + str(freq_max) + " Hz")
    fft = scipy.fftpack.rfft(data, axis=axis)
    frequencies = scipy.fftpack.fftfreq(data.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - freq_min)).argmin()
    bound_high = (np.abs(frequencies - freq_max)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0

    result = np.ndarray(shape=data.shape, dtype='float')
    result[:] = scipy.fftpack.ifft(fft, axis=0)
    result *= amplification_factor
    return ntsc2rgb(result)
