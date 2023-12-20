# SampleGen: AI generation of Audio Samples

## Description

The overall goal is to be able to implement a sythesizer that can generate new sounds in an interesting manner without the user having to tweak 100s of parameters.

The synthesizer itself would be written as an audio plug-in, typically written in C++ using libraries such as JUCE and released as AU and VST.

Machine Learning hopefully provides an opportunity to create interesting new wave forms. The google NSynth released in 2016 aimed to do something similar, this leverages the WaveNet research.

My proposal here is to create a deep-learning model that can regenerate audio samples of musical instruments reasonably faithfully. The model should then be able to generate new samples, either by interpolation between samples, or randomly. The "reasonably faithfully" is important, the goal is not to replicate existing sample libraries, but rather to generate new and interesting sounds.


Please see README.pdf for a more complete description of the project and its implementation.


