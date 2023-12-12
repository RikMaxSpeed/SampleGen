# SampleGen: AI generation of Audio Samples

## Description

The overall goal is to be able to implement a sythesizer that can generate new sounds in an interesting manner without the user having to tweak 100s of parameters.

The synthesizer itself would be written as an audio plug-in, typically written in C++ using libraries such as JUCE and released as AU and VST.

Machine Learning hopefully provides an opportunity to create interesting new wave forms, see the google NSynth released in 2016 which leverages the WaveNet research.

My proposal here is to create a deep-learning model that can regenerate audio samples of musical instruments reasonably faithfully. The model should then be able to generate new samples, either by interpolation between samples, or randomly. The "reasonably faithfully" is important, the goal is not to replicate sample libraries, but rather to generate new and interesting sounds.

## Data

The model is trained on a dataset of public domain samples acquired from websites such as https://freewavesamples.com.

Currently approx 950 samples have been gathered, when restricted purely to middle-C. Interpolation at other frequencies is currently out of scope.

The samples cover many musical instruments, both acoustic, and electronic, including a large number of synthesized sounds. 

If needed, the data can be augmented by simply mixing 2 samples, other means may be possible too, but it's important not to distort the spectrum features that we're trying to model.


## Implementation

### Spectograms vs Audio Samples

When working with audio and digital signal processing, an immediate question is whether to work in the time domain with audio samples, or in the frequency domain using the short-time fourier transform (STFT) for example.

Algorithms such as the seminal 2016 WaveNet paper by google work in the audio sample space - this has since been significantly improved, and the current state-of-the-art is to work in using audio samples to generate ultra-realistic speech or vocal synthesis.

However working with samples, at 44.1kHz, requires a huge amount of compute, and it is also more difficult to interpret. The loss function would probably require an STFT to make sense.

Therefore I decided for a first foray to work using spectrogram and specifically the STFT.

Other frequency spectrum representations could be considered:
- Mel Frequency Cepstrum Coefficient (MFCC) could be considered, however these are not suitable for audio reconstruction.
- Constant-Q transforms: these could be useful for analysing music itself where precise note onset and end times are necessary. However this is not the scope of the current project.

### An early compromise

In a first pass I discarded all phase information, using only the amplitudes. Whilst this reduces the data-size by half, and considerably simplifies the interpretation, it did introduce significant artefacts, even if replaying the original data. Ultimately I switched to using both the real & imaginary parts of the STFT, and whilst this doubled the amount of data to be modelled, it did significantly reduce the systemic artefacts dues to zeroing the phases.


### Auto-Encoder

An auto-encoder presents itself as a natural choice as we are interested in both sample recreation and new sample generation. The auto-encocder should be able to represent each of the source samples as a unique vector of numbers in a latent space. New samples can then be created in multiple ways: interpolating between samples in that space, or randomly perturbing the vector coordinates of a given sample, or simple generating a random vector.

### Variational Auto-Encoder (VAE)

After some exploration with simple models, I bumped into some well documented problems: 
- when interpolating between encoded samples, I would frequently get silence (!!), and occasionally horrible noise. 
- the variables in the control dimensions had no definite range, which also made it difficult to generate new random samples.

This is because the auto-encoded values have no constraints over them, random values might simply point to regions of emptiness which the model has never seen before.

These problems can be resolved using a Variational Auto-Encoder. The encoder effectively produces a distribution of possible encodings, represented as a vector of means and standard deviations - a random sample can be selected from these distributions to represent a given sound. The decoder then learns these distributions and uses this information to recreate the original sample.

Further constraints are imposed on the encoder: the means for each encoded variable should be 0, and the standard deviation 1. This helps ensure that the individual encoded values stay within a reasonable interval, and facilitates interpolation or using the model as as generator.

The mathematics of the VAE are reasonably complex, using Kullback-Leibler divergence (https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) to estimate how far the model's distribution is from an ideal canonical normal distribution with mean 0 and variance 1. In practice the literature provides various tricks to implement this, in particular using the log variance of the distribution. Fortunately I found many online tutorials and videos explaining the concepts and how to implement a VAE.

### Evolution of the Model

My initial auto-encoder model was simply four fully-connected layers, the decoder is implemented as a reverse of the encoder. This allowed me to verify that my entire tech stack was working. Some experimenting revealed that three layers were unecessary, however four could achieve reasonably good accuracy.
 
I then started looking at models that could interpolate across the frequency spectrum and across time in a more reasonable manner.

1: MLP: at each time-step, an MLP is trained using the previous spectrogram slice (initialised to 0 for the first slice) and the current slice, it outputs a vecto of "control" variables. The decoder is a symmetric version of this process, taking in the prevoius generated slice (0s initially) and the control variables, and generates the corresponding spectrogram slice. 

2: RNN: in a similar way, an RNN is trained at each time step, learning how best to summarise 1 time-step into a set of control variables.

In practice the RNN is maybe 2 to 3x faster than the MLP and can reach similer accuracy, but with a larger model.


3: Variational Auto-Encoder: this is an inner-core that maps from the hidden variables per time-step, to a small latent space. It has not been implemented in such a way that it's reasonably easy to use it inside any other naive Auto-Encoder, for example combining the MLP Auto-Encoder or the RNN Auto-Encoder to create VAE versions of these with much smaller latent spaces.

4: GRUs, LSTMs, ...: further models could be explored with better modelling of the evolution of the spectrum over time.

## Training & Hyper-Parameters

Training has proven extremely difficult. As we already know from the practical experience gathered during this course, hyper-parameter tuning is paramount.

To keep things simple, I used skopt's gp_minimise (https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html). Internally this uses a Matern kernel  + Gaussian noise, and provides a configurable search-space for integers, reals and categories, with uniform or log distributions. Generating plots of the hyper-parameter tuning over time has been helpful to prove that it generally works.

Lessons & Obeservations:

- Possibly due to the complexity and size of the training data-set, the hyper-parameter tuning always veers towards the smallest batch sizes allowed. This makes training very slow as we're no longer able to properly leverage the available GPU power (would require multi-threading the GPR).

- It's important to set limits on the model-size as the hyper-parameter optimisation can easily veer to multi-gigabyte models!

- In order to be able to train models with different numbers of layers, I've created a function 'interpolate_layer_sizes(start, end, depth, ratio)'. Start and end are usually determined by the problem set, leaving us with 2 hyper-parameters, depth = the number of layers, ratio = the power to use whilst interpolating. This is reasonably flexible, and allows me to create models where the layer sizes are large at the beginning, or the end, or flat.

- I don't believe it's possible to reduce the training data-set size when optimising hyper-parameters, because the purpose of this model is to learn as many wave forms as possible. If the training data-set is shrunk, a smaller model may be found by the hyper-parameter tuning that wouldn't then be able to learn the larger data-set.

- Stopping conditions can be complex: the Adam optimiser sometimes jumps to a significantly worse loss, then recovers to an overall better loss. I therefore stop if the moving average of the loss is stalled.

- The hyper-parameter optimiser seems minimise the the weight-decay. Somehow this concept doesn't appear useful in this use-case.


## Over-fitting

Whilst training I output an overfitting ratio: train loss / test loss. This is helpful to identify whether the model is beginning to overfit.

However, for this audio generation problem, it would be nice to be able to reproduce important samples as accurately as possible, so overfitting the original training samples may not necessarily be a bad thing.

But over-fitting could also lead to the Auto-Encoder being less able to generate diverse outputs outside the original training dataset. 


## Incremental Training

The models I'm using are a combination of 2 models: a naive auto-encoder on the outer layer, followed by a VAE. The outer layer typically compresses the data by a factor of 6. The inner VAE layer compresses the layer by a factor of 3000 to 4000, down to a small number of latent variables.

In practice, training the two models simultaneously proves intractable:
Naive-Encoder -> VAE-Encoder -> Latent Space -> VAE-Decoder -> Naive-Decoder

The model is simply too deep with possibly over 15 layers. The hyper-parameter tuning space is also the product of the spaces for each individual model.

As the performance of the end-to-end VAE will be gated by the performance of the outer naive encoder, I decided to split the training as follows:
1. Find the best hyper-parameters for the outer "naive" auto-encoder.

2. Train the "naive" outer auto-encoder with longer time scales.

3. Hyper-train train the VAE part of the full auto-encoder using a frozen set of optimal parameters for the outer auto-encoder.

4. Train the full auto-encoder with longer time-scales.

This has several benefits:

- We can optimally tune the outer auto-encoder, with the RNNs, MLPs, LSTMs etc.

- We can then tune the VAE whilst training the entire model end-to-end.

- Importantly: the end-to-end model training would fail because it was too complicated (disappearing gradients etc.)

- The search space for the hyper parameter tuning is significantly reduced. If the outer model had a search-space of N parameters, and the inner VAE had a search space of M, the original search-space would be N * M, whilst it is now N + M. (ie: 9 vs 6 in my use-case)

- We can use different optimiser hyper-parameters (learning rate, batch size, weight decay) for the VAE than for the outer auto-encoder.


## Tests

### Sound Quality

Identifying a suitable loss metric for audio is complex. Converting to a log scale such as decibels would be a reasonable first pass, but that would further complicate the gradients in the complex STFT case.

Ultimately I'm simply using the MSE Loss between the original and regenerated spectograms, and whilst this has no human perceptual meaning, it does allow the models to converge reasonably.

An option I did explore but haven't re-enabled is using mu-law encoding. This would enable the model to prioritise loud parts of the sound vs quiter ones (similarly to using a decibel scale). However, it's also important to capture the "decay" or "tail-off" portions of sounds, the amplitudes can scale down by a factor of 100, which to the human ear is just the same sound but not as loud.

Ultimately the best test is simply human perception, and at present the models don't sound that great anyway, so this isn't really an tough issue.
  

### Artefacts

Whilst I originally thought to work purely in the Magnitude (amplitude) space, which also allowed for a decibel-conversion and loss calcuations, it turned out the artefacts this introduced were unnacceptable. I therfore doubled the model size, using the complex STFT, scaled to magnitude 1, so the real & imaginary parts are all in the range [-1, 1] which is also helpful for the models.

At this point there are no systemic artefacts introduced by the data representation itself, this can be verified by transforming a sample to an STFT, then normalising it, the de-normalising it, converting it back to a complex STFT, amplifying it and playing the sound (as well as displaying the reconstituted spectogram.

All artefacts are therefore a consequene of the model's own encoding.

Using a custom loss function (currently disabled) could help focus the model's attention on the critical parts of the audio spectrum, and also the critical time: human's identify sounds primarily through their attack section (ie: the first 250 msec).


### VAE Distribution

I've created plots of all the variables in the latent space: individually, in pairs, and for selected sub-sets of samples types, for example pianos vs strings.

This has been helpful to highlight the size of the latent space required (sometimes a variable will be barely used), and that generally we are achieving our ideal mu=0, std=1 distribution for each variable.  


## Key Lessons Learnt

The key points here are very similar to my work on the CapStone hyper-parameter optimisation problem:

- You need to know in detail what's the code is doing. I have stumbled on so many gotchas which can be sometimes masked by the capabilities of the deep neural networks used. For example, it took me a long time to resolve an artefact that was introduced by the layout of the real & imaginary numbers in memory.

- ChatGPT-4 is an invaluable help as a way of protoyping code, or getting advice on how to approach problems.

- Data Visualisation is really important: this helps spot so many problems, or prove that things are working as you'd expect them to.

- Variational Auto-Encoders are a necessity: I quickly (re-)discovered why naive auto-encoders were not sufficient.

- It's unnecessary to add a with to the KL-Divergence term in the VAE loss function: it seems the optimiser is always able to force the distributions into mu=0, stdev=1.

- YouTube tutorials can also be very helpful. There are long series (which I've partially watched) on audio synthesis in particular, music generation etc.

- Working with Audio is significantly different from working with Images. CNNs do not work well with Audio, nor does max-pooling apply well. An interesting article on Medium explained that audio is "transparent", unlike objects in an image which occlude each other. If you play two sounds together, you hear two sounds (in most cases), one sound does not hide the other, whilst in images each pixel can generally be attributed to a single object. There are also complex scaling issues: a sound at a 100x smaller magnitude is still perceived as the original sound. The human ear can accuratley hear sounds spanning over 100 dB in range, ie: 5 orders of magnitude!

## Future work

The key item to work on is better modelling in the time-domain. I don't know whether LSTMs or some other transformer model will crack this, but this is the current gating factor.

Increasing the sample dataset would also be helpful.

Ideally I would like to shift to working using audio samples rather than spectograms, that appears to be what is used in all the state-of-the-art models and publications.


## Conclusion

This has been an huge and time-consuming project. It woudln't have been possible without having my own GPU-accelerated hardware. The overall results are sadly not as good as I would have liked and will require more work. Overall, Whilst I've learnt a lot implementing all of this myself, I've also got a much better appreciation of how much more I need to learn!


