#  SampleGen: AI modelling of Audio Samples

## Description

The overall goal is to be able to implement a sythesizer that can generate new sounds in an interesting manner without the user having to tweak 100s of parameters.

The synthesizer itself would be written as an audio plug-in, typically written in C++ using libraries such as JUCE and released as AU and VST.

Machine Learning hopefully provides an opportunity to create interesting new wave forms, see the google NSynth released in 2016 which leverages the WaveNet research.

My proposal here is to reate an AI model that can reasonably faithfully regenerate audio samples of musical instruments. The model should then be able to generate new samples, either by interpolation between samples, or randomly.

## Data

The model is trained on a dataset of public samples acquired from websites such as https://freewavesamples.com.

Currently approx 980 samples have been gathered, restricted purely to the note C4 (interpolation at other frequencies is currently out of scope).

If needed, the data can be augmented by simply mixing 2 samples.

## Implementation

### Spectograms vs Audio Samples

When working with audio and digital signal processing, an immediate question is whether to work in the time domain with audio samples, or in the frequency domain using the short-time fourier transform (STFT) for example.

Algorithms such as the seminal 2016 WaveNet paper by google work in the audio sample space - this has since been significantly improved, and the current state-of-the-art is to work in using audio samples to generate ultra-realistic speech or vocal synthesis.

However working with samples, at 44.1kHz, requires a huge amount of compute, and is also more difficult to interpret.

Therefore I decided for a first foray to work using spectrogram and specifically the STFT.

Other frequency spectrum representations could be considered:
- Mel Frequency Cepstrum Coefficient (MFCC) could be considered, however these are not suitable for audio reconstruction.
- Constant-Q transforms: these could be useful for analysing music itself where precise note onset and end times are necessary. However this is not the scope of the current project.

### A big compromise

In the current implementation, I am discarding all phase information from the STFT, and only using the amplitudes. Whilst this reduces the data-size by half, and considerably simplifies the interpretation, it does introduce noticable artefacts, even if replaying the original data.
  
It might be possible in future, with a more sophisticated model, to capture the phase information and use this to reconstruct the signal more accurately.


### Auto-Encoder

An auto-encoder presents itself as a natural choise as we are interested in both sample recreation and new sample generation. The auto-encocder should be able to represent each of the source samples as a unique vector of numbers in a latent space. New samples can then be created in multiple ways: interpolating between samples in that space, or randomly perturbing the vector coordinates of a given sample, or simple generating a random vector.

### Variational Auto-Encoder

After some exploration with simple models, I bumped into some well documented problems: when interpolating between encoded samples, I would frequently get silence (!!), and occasionally horrible noise.

This is because the auto-encoded values have no constraints over them, random values might simply point to regions of emptiness which the model has never seen before.

These problems can be resolved using a Variational Auto-Encoder. The encoder effectively produces a distribution of possible encodings, represented as a vector of means and standard deviations - a random sample can be selected from these distributions to represent a given sound. The decoder then learns these distributions and uses this information to recreate the original sample.

Further constraints are imposed on the encoder: the means for each encoded variable shoudl be 0, and the standard deviation 1. This helps ensure that the encoded values stay within a reasonable interval.

The mathematics of the VAE are reasonably complex, using Kullback-Leibler divergence (https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) to estimate how far the model's distribution is from an ideal canonical normal distribution with mean 0 and variance 1. In practice the literature provides various tricks to implement this, in particularl using the log variance of the distribution. I was easily able to find multiple online tutorials explaining how to implement this!

### The Model

My initial model was simply four fully-connected layers, the decoder is implemented as a reverse of the encoder. This allowed me to verify that my entire tech stack was working. Some experimenting revealed that three layers were unecessary, however four could achieve reasonably good accuracy.
 
 

## Tests

### Sound Quality

### Artefacts

### VAE Distribution

### Hyper-Parameter Optimisation


## Key Lessons Learnt

VAE 


## Conclusion


