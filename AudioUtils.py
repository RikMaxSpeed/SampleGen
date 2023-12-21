import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import sounddevice as sd
import soundfile as sf
import torch
import math
import os
import time

import Device
from Debug import debug

from Graph import PlotVideoMaker


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


middleCHz = note_frequency(60) # Middle-C
print(f"middle-C={middleCHz:.2f} Hz")
assert(abs(middleCHz - 261.63) < 0.1)

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
    


stft_video = PlotVideoMaker("STFT_Video", True, 0.5)

def save_stft_video():
    global stft_video
    stft_video.automatic_save()
    
    
def start_new_stft_video(name, auto_save):
    global stft_video
    stft_video.automatic_save() # save anything that might be outstanding.
    stft_video = PlotVideoMaker(name, auto_save, 0.5)



def plot_stft(name, stft_result, sr, hop_length):
    """Plots the STFT"""
    # Convert amplitude to dB for visualization
    dB = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(name)
    plt.tight_layout()
    
    stft_video.add_plot(True)


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

    debug("stft", stft)
    
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


def amplitude_to_dB(amplitude):
    return 20.0 * torch.log10(torch.clamp(amplitude, min=1e-5))  # Added clamp to avoid log(0)

def dB_to_amplitude(dB):
    return 10.0 ** (dB / 20.0)


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



class AmplitudeCodec:
#    def __init__(self):

    def encode(self, x):
        return x.abs()

    def decode(self, y):
        stacked = torch.stack([y, torch.zeros_like(y)], dim=-1)
        return torch.view_as_complex(stacked)


def testAmplitudeCodec():
    print("\n\n\nTesting Amplitude conversion\n")
    codec = AmplitudeCodec()
    
    complex = torch.rand(5, 3, dtype=torch.complex32)
    print(f"complex={complex}")
    
    amplitudes = codec.encode(complex)
    print(f"amplitudes={amplitudes}")

    decode = codec.decode(amplitudes)
    print(f"decode={decode}")
    
    recode = codec.encode(decode)
    print(f"recode={recode}")
    
    diff = (recode - amplitudes).abs()
    print(f"diff={diff}")


if __name__ == '__main__':
    testAmplitudeCodec()


# We normalise all amplitudes to [0, 1] on input.
# But when we convert back to audio we need to amplify the signal again.
maxAmp = 278.0 # Average across all training samples - this is just to get a reasonable playback level.

def complex_to_mulaw(complex_tensor):
    global maxAmp
    
    magnitude = torch.abs(complex_tensor)
    
    max = torch.max(magnitude)
    
    return codec.encode(magnitude / max)
    
    
def mulaw_to_zero_phase_complex(mulaw_tensor):
    # Create a complex tensor with the linear amplitude as the real part and 0 as the imaginary part
    magnitude = codec.decode(mulaw_tensor) * maxAmp
    #magnitude[magnitude < maxAmp/100] = 0 # hack! remove noisy low values
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
    
    
if __name__ == '__main__':
    test_mulaw_conversion()


def complex_to_normalised_polar(complex_tensor):
    amplitudes = torch.abs(complex_tensor)
    phases = torch.angle(complex_tensor)

    # normalise
    amplitudes /= torch.max(amplitudes)
    phases /= torch.pi

    return torch.stack((amplitudes, phases), dim=-1)

def normalised_polar_to_complex(interleaved_tensor):
    debug("normalised_polar_to_complex", interleaved_tensor)
    amplitudes = interleaved_tensor[..., 0] * maxAmp
    debug("amplitudes", amplitudes)
    phases = interleaved_tensor[..., 1] * torch.pi
    debug("phases", phases)
    return torch.polar(amplitudes, phases)


def test_complex_to_polar_and_back():
    print("\n\n\nTesting Complex <--> Polar\n")
    N = 10
    magnitudes = torch.rand(N) * 100
    phases     = (torch.rand(N) - 0.5) * 2 * torch.pi
    #phases = torch.zeros(N)

    complex = magnitudes * torch.exp(1j * phases)
    print(f"complex={complex}")
    amp = torch.max(torch.abs(complex))
    print(f"amp={amp:.3f}")
    
    polar = complex_to_normalised_polar(complex)
    print(f"polar={polar}")
    debug("polar", polar)
    
    back = normalised_polar_to_complex(polar)
    print(f"back={back}")
    debug("back", back)
    
    back *= amp / maxAmp
    print(f"scaled={back}")
    
    delta = back - complex
    norm = torch.norm(delta, p=2)
    print(f"delta={delta}, norm={norm:.5f}")
    
    assert(norm < 1e-4) # we expect some numeric noise with float32


if __name__ == '__main__':
    test_complex_to_polar_and_back()


# Minimise the flutter when converting from a magnitude spectogram back to audio
def recover_audio_from_magnitude(magnitude_spectrogram, stft_size, stft_hop, sample_rate, iterations=32):
    # Griffin-Lim phase reconstruction
    magnitudes = magnitude_spectrogram.detach().cpu().numpy()
    
    audio = librosa.griffinlim(magnitudes, n_iter=iterations, hop_length=stft_hop, win_length=stft_size)


    # We convert back to STFT to ease integration with the rest of the code!!
    return compute_stft(audio, sample_rate, stft_size, stft_hop)


import subprocess
import platform

def is_running_on_mac():
    return platform.system() == 'Darwin'

def say_out_loud(text):
    if is_running_on_mac():
        subprocess.run(['say', '-v', 'Reed', '-r', '150', text], check=True)
    else:
        print(f"TTS is not implemented for {platform.system()}: '{text}'")

if __name__ == '__main__':
    say_out_loud("Testing 1, 2, 3.")
