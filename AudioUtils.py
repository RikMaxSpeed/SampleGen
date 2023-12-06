import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import sounddevice as sd
import soundfile as sf
import torch
import math
import os
from Device import *
from Debug import *



def is_power_of_2(n):
    return n > 0 and (n & (n - 1)) == 0


def linterp(t, a, b):
    return a + t * (b - a)


def convert_to_float(data):
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    
    return data


def convert_to_mono(data):
    if len(data.shape) == 1:
        return data
        
    channels = data.shape[1]
    if channels == 1:
        return data
        
    #print("converting stereo to mono")
    data = (data[:, 0] + data[:, 1]) / 2
    return data


def normalise(data):
    peak = np.max(data)
    #print("peak={:.3f}".format(peak))
    if peak > 0:
        data /= peak
    
    return data


def note_frequency(midi):
    return 440 * 2 ** ((midi - 69) / 12) # 440Hz = A4 = 69


c4hz = note_frequency(60) # Middle-C


def normalise_sample_to_mono_floats(data):
    data = convert_to_float(data)
    data = convert_to_mono(data)
    data = normalise(data)
    
    #debug("normalise_sample_to_mono_floats", data)

    return data


def read_wav_file(file_name):
    """Reads a .wav file and returns the sample rate and data"""
    sr, data = wavfile.read(file_name)
    #print("loaded file={}, sample rate={} Hz, data={} x {}".format(file_name, sr, type(data.dtype), data.shape))
    return sr, data


def compute_stft(data, sr, n_fft, hop_length, win_length=None, window='hann'):
    """Computes the STFT of the audio data"""
    return librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    

def plot_stft(name, stft_result, sr, hop_length):
    """Plots the STFT"""
    # Convert amplitude to dB for visualization
    dB = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(name + f" ({sr} Hz)")
    plt.tight_layout()
    plt.show()


def save_to_wav(file_name, data, sr):
    """Saves the data to a .wav file"""
    data_int16 = (data * 32767).astype(np.int16)
    wavfile.write(file_name, sr, data_int16)


def istft_to_audio(stft, hop_length, win_length=None, window='hann'):
    """Computes the inverse STFT to get the audio signal back"""
    #debug("istft_to_audio", stft)
    assert(stft.shape[0] % 2 == 1)
    
    return librosa.istft(stft, hop_length=hop_length, win_length=win_length, window=window)


def play_audio(data, sr):
    """Plays the audio data using sounddevice"""
    sd.play(data, sr)
    sd.wait()  # Wait for the audio to finish playing


def save_and_play_audio_from_stft(stft, sr, hop_length, write_to_file, playAudio):
    # Compute inverse STFT to reconstruct the audio
    #debug("stft", stft)
    audio = istft_to_audio(stft, hop_length)
    #debug("audio", audio)
    #print("reconstitued audio={} samples at {} Hz, duration={:.1f} sec".format(len(audio), sr, len(audio)/sr))

    # Save the reconstructed audio to a new .wav file
    if write_to_file is not None:
        save_to_wav(write_to_file, audio, sr)

    # Play the reconstructed audio
    if playAudio:
        play_audio(audio, sr)


def compute_stft_for_file(file_name, n_fft, hop_length):
    sr, data = read_wav_file(file_name)
    #print("sr={} Hz, duration={:.1f} sec, hop={} -> {:.1f} windows".format(sr, len(data)/sr, hop_length, len(data)/hop_length))
    
    data = normalise_sample_to_mono_floats(data)
    #debug("normalised", data)
    
    stft = compute_stft(data, sr, n_fft, hop_length)
    #debug("compute_stft", stft)
    return sr, stft


def demo_stft(file_name, n_fft, hop_length):
    sr, stft = compute_stft_for_file(file_name, n_fft, hop_length)
    
    plot_stft(file_name, stft, sr, hop_length)
    
    save_and_play_audio_from_stft(stft, sr, hop_length, "Results/resynth-" + os.path.basename(file_name), True)
    

def print_default_audio_device_info():
    default_device = sd.default.device
    device_info = sd.query_devices(device=default_device)

    print("Default Sound Device Info:")
    for key, value in device_info.items():
        print(f"{key}: {value}")
        

def print_all_audio_devices_info():
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        print(f"Device #{index + 1}")
        for key, value in device.items():
            print(f"{key}: {value}")
        print("-" * 50)


#def amplitude_to_dB(amplitude):
#    return 20.0 * torch.log10(torch.clamp(amplitude, min=1e-5))  # Added clamp to avoid log(0)
#
#def dB_to_amplitude(dB):
#    return 10.0 ** (dB / 20.0)


## WaveNet uses Mu-Law encoding which is simlar but constrained to the range [-1, 1].
## So let's normalise the results to [0, 1] = min amplitude, max amplitude
#
#min_dB = -60
#
#def complex_to_dB_amplitude(complex_tensor):
#    # Compute the magnitude
#    magnitude = torch.abs(complex_tensor)
#
#    threshold = min_dB
#    result = amplitude_to_dB(magnitude)
#    
#    # Remove low-volume data:
#    result[result < min_dB] = min_dB
#    
#    return 1 - (result / min_dB) # normalise
#
#    
#def dB_amplitude_to_zero_phase_complex(dB_tensor):
#    dB_tensor = (1 - dB_tensor) * min_dB
#    
#    result = dB_to_amplitude(dB_tensor)
#
#    # Create a complex tensor with the linear amplitude as the real part and 0 as the imaginary part
#    #return torch.complex(result, torch.zeros_like(result).to(device)) # not supported on MPS!
#    stacked = torch.stack([result, torch.zeros_like(result).to(device)], dim=-1)
#    return torch.view_as_complex(stacked)
#    
#    
#def test_dB_conversion():
#    a=0.1
#    real_part = torch.tensor([1.0, 0.5, 0.25, 0.0])
#    imaginary_part = torch.tensor([a, a, a, a])
#    complex_tensor = torch.complex(real_part, imaginary_part)
#    print("       signal=", complex_tensor)
#    amps = complex_to_dB_amplitude(complex_tensor)
#    print("         amps=", amps)
#    resynth = dB_amplitude_to_zero_phase_complex(amps)
#    print("      resynth=", resynth)
#    amps2 = complex_to_dB_amplitude(resynth)
#    print("amps(resynth)=", amps)
#    print("         diff=", amps2-amps)


class MuLawCodec:
    def __init__(self, mu):
        self.mu = mu

    def mu_tensor(self, x):
        return torch.tensor(self.mu, dtype=x.dtype)
        
    def encode(self, x):
        x = torch.clamp(x, -1, 1)
        mu = self.mu_tensor(x)
        magnitude = torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
        return torch.sign(x) * magnitude

    def decode(self, y):
        mu = self.mu_tensor(y)
        magnitude = (torch.exp(torch.abs(y) * torch.log1p(mu)) - 1) / mu
        return torch.sign(y) * magnitude


codec = MuLawCodec(2) # Yet another hyper-parameter but we can't tune this one as it's outside the model's loss function.


# We normalise all amplitudes to [0, 1] on input.
# But when we convert back to audio we need to amplify the signal again.
maxAmp = 278 # Average across all training samples - this is just to get a reasonable playback level.

def complex_to_mulaw(complex_tensor):
    global maxAmp
    
    magnitude = torch.abs(complex_tensor)
    
    max = torch.max(magnitude)
    
#    if max > maxAmp:
#        maxAmp = max
#        print("*** warning: max={:.1f} --> {:.1f} !!".format(max, maxAmp))

    return codec.encode(magnitude / max)
    
    
def mulaw_to_zero_phase_complex(mulaw_tensor):
    # Create a complex tensor with the linear amplitude as the real part and 0 as the imaginary part
    magnitude = codec.decode(mulaw_tensor) * maxAmp
    magnitude[magnitude < maxAmp/100] = 0 # hack! remove noisy low values
    stacked = torch.stack([magnitude, torch.zeros_like(magnitude).to(magnitude.device)], dim=-1)
    return torch.view_as_complex(stacked)


def test_mulaw_conversion():
    print("\n\n\nTesting Mu-Law conversion\n")
    a=0.1
    real_part = torch.tensor([1.0, 0.5, 0.25, 0.0, -0.7, -0.9])
    imaginary_part = torch.full(real_part.shape, a)
    complex_tensor = torch.complex(real_part, imaginary_part)
    print("       signal=", complex_tensor)
    amps = complex_to_mulaw(complex_tensor)
    print("         amps=", amps)
    resynth = mulaw_to_zero_phase_complex(amps)
    print("      resynth=", resynth)
    amps2 = complex_to_mulaw(resynth)
    print("amps(resynth)=", amps)
    print("         diff=", amps2-amps)

    plt.figure(figsize=(9, 4))

    for n in (0, 1, 2, 4, 8, 16, 24):
        mu = 2**n
        c = MuLawCodec(mu)
        Xs = torch.tensor([x for x in np.arange(-1.1, 1.1, 1/100)])
        Ys = c.encode(Xs)
        plt.plot(Xs, Ys, label = f"mu=2^{n}")
#        Zs = c.decode(Ys)
#        plt.plot(Xs, Zs, label = f"mu=2^{n}")
        
    plt.title("Mu-Law")
    plt.legend()
    plt.show()
    
    
#test_mulaw_conversion()

