Audio
Samples
Single Note
Acoustic
Synthesised


# Dataset Card for Audio Samples

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-instances)
  - [Data Splits](#data-instances)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)

## Dataset Description

- **Homepage:** [Needs More Information]
- **Repository:** https://github.com/RikMaxSpeed/SampleGen
- **Paper:** [Needs More Information]
- **Leaderboard:** [Needs More Information]
- **Point of Contact:** RikMaxSpeed@gmail.com

### Dataset Summary

A collection of approx 1000 audio samples of single notes from acoustic or synthesised instruments, containing only the notes C3 and C4.

This data-set is intended to enable the generation of new samples, which  can then be performed by a synthesiser.

The file names, which typically describe the samples, are in English.



### Supported Tasks and Leaderboards

Imperial personal College CapStone project.

### Languages

English, colloquial, with some inside musical industry jokes, for example spelling the famous "Rhodes" piano as "Roadz".


## Dataset Structure

### Data Instances

All files are in the .wav format, the majority have a sample rate of 44.1 kHz and a few are at 48 kHz.

### Data Fields

None other than the file name from which a sample type or category could potentially be inferred.

### Data Splits

The data-split is up to the user. Typically train=80%, test=20%.

## Dataset Creation

### Curation Rationale

The data-set was extracted from multiple sources, including:
- Free Wave Samples: public domain wave samples https://freewavesamples.com
- Music Radar: https://www.musicradar.com/news/sampleradar-free-essential-synth-samples
- Apple Logic (licensed software)
- Arturia Pigments (licensed software)

Although single-note samples are typically not copyrightable, these are all samples for which I, personally, hold a license.

This data-set is therefore not in the public-domain.


### Source Data

#### Initial Data Collection and Normalization

The only normalisation here was to restrict the files to those containing the note C3 or C4.
This could be expanded in future.

#### Who are the source language producers?

File names are in English. No further text is available.

### Annotations

#### Annotation process

None.

#### Who are the annotators?

Not Applicable

### Personal and Sensitive Information

These audio files contain no personal information.
However some come from licensed software distributions, therefore this dataset is  for my personal use only.



## Considerations for Using the Data

### Social Impact of Dataset

None.

### Discussion of Biases

These are single-note samples of musical instruments from all over the world. 
However there is a strong bias towards instruments used in Western music, ie: piano, strings, brass, organ etc.
Note: this dataset does not include audio-loops or sound effects.


### Other Known Limitations

[Needs More Information]

## Additional Information

### Dataset Curators

Richard Meyer

### Licensing Information

Parts of the dataset are sourced from Apple & Arturia, therefore this dataset is not publicly available.

### Citation Information

[Needs More Information]