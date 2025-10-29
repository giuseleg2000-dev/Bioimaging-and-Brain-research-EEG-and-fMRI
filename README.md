# Bioimaging and Brain Research: EEG and fMRI Analysis
This repository contains two comprehensive studies exploring brain function through complementary neuroimaging approaches: electroencephalography (EEG) and functional magnetic resonance imaging (fMRI).
Repository Overview
This project presents independent analyses of brain activity using different modalities:

* EEG Study: Investigation of music-induced emotions through engagement indices
* fMRI Study: Identification of resting-state networks through Independent Component Analysis

Studies
1. EEG Study: Emotional Engagement During Music Listening
Objective
Identify EEG-derived engagement indices that effectively discriminate between:

Resting state vs. music listening
Positive vs. negative emotions
Specific pairs of music-induced emotions

Dataset

Source: OpenNeuro dataset "An EEG dataset recorded during affective music listening" (DOI: 10.18112/openneuro.ds002721.v1.0.2)
Participants: 31 healthy adults (analyzed subjects 12-21)
Recording system: BrainAmp EEG amplifier with 19 electrodes (10-20 system)
Sampling rate: 1000 Hz (resampled to 500 Hz)
Sessions: 6 runs per subject (2 resting-state, 4 music listening)

Methodology
Preprocessing Pipeline:

Bandpass filtering (0.5-80 Hz, 3rd order Butterworth)
Power line noise removal (50 Hz, using Cleanline plugin)
Average reference re-referencing
Independent Component Analysis (ICA) for artifact removal
ICLabel-based component classification

Analysis:

37 engagement indices computed from frequency band ratios (δ, θ, α, SMR, β, γ)
Wilcoxon signed-rank test for statistical comparison
Emotion classification based on Russell's Circumplex Model of Affect

Key Findings
Most Discriminative Indices (Resting vs. Music):

I1, I4, I5, I7, I11, I12, I19, I20, I22, I24, I25, I26, I27, I28, I30, I35, I36

Most Significant Channels:

Frontal: FP1, FP2, F4, F8
Central: T3, C3, CZ
Parietal: PZ
Occipital: O1, O2, T6

Emotion Discrimination:

Pleasant vs. Unpleasant emotions best distinguished by channels: FP1, F8, FP2, PZ, O1, P3
Key indices for emotion pairs: I1, I2, I7, I14, I20, I23, I24, I28, I30

Notable Engagement Indices:

I1 (β/α): Reflects engagement and alertness
I20 (α/γ): Evaluates relaxation vs. cognitive load
I5 (θ/δ): Indicates deep relaxation or drowsiness
I4 (θ/α): Associated with creativity and relaxed conscious states
I7 (SMR/β): Distinguishes tension from relaxation


2. fMRI Study: Resting-State Network Identification
Objective
Identify and classify resting-state networks (RSNs) and noise components from rs-fMRI data using manual ICA component classification.
Dataset

Source: Yale Resting State fMRI/Pupillometry: Arousal Study (OpenNeuro)
Scanner: Siemens 3T Prisma with 64-channel coil
Structural imaging: 3D MPRAGE (1×1×1 mm³, TR=2400ms)
Functional imaging: 2D multiband EPI (2×2×2 mm³, TR=1000ms, 410 volumes)

Methodology
Preprocessing (FSL 6.0/FEAT):

Brain extraction (BET, fractional intensity=0.33)
Motion correction (MCFLIRT)
Temporal filtering (high-pass 0.01 Hz)
Spatial smoothing (6mm FWHM Gaussian kernel)
MNI152 2mm standard space normalization

ICA Analysis (MELODIC):

Automatic dimensionality estimation
Z-score threshold: 0.35
Manual component classification following Griffanti et al. guidelines
Validation against Smith et al. canonical RSN templates

Classification Criteria
Signal Components:

Large, coherent grey matter clusters
Smooth time series
Low-frequency dominated power spectra (<0.1 Hz)
Anatomically plausible patterns

Noise Components:

Fragmented/scattered spatial patterns
Located in white matter, CSF, or peripheral regions
Abrupt temporal spikes or drifts
Atypical spectral profiles

Results
Identified Resting-State Networks (10 components):

Medial visual area
Occipital pole
Lateral visual area
Default mode network (DMN) - precuneus, posterior cingulate, inferior parietal lobules, ventromedial PFC
Cerebellar network
Sensorimotor network - supplementary motor area, primary sensorimotor cortex
Auditory network - superior temporal gyrus, posterior insula
Medial frontal network - anterior cingulate, paracingulate cortex
Left frontoparietal network
Right frontoparietal network

Noise Components Identified:

Motion artifacts (cortical edge rings)
Physiological noise (cardiac/respiratory fluctuations in brainstem)
Vascular artifacts
Scanner-related noise


Tools and Software
EEG Analysis

MATLAB with EEGLAB toolbox
Plugins: Cleanline, ICLabel
Statistical testing: Wilcoxon signed-rank test

fMRI Analysis

FSL 6.0 (FMRIB Software Library)

FEAT (preprocessing)
MELODIC (ICA)
FSLeyes (visualization)


Standard space: MNI152 2mm template

Repository Structure
├── EEG_Analysis/
│   ├── preprocessing/
│   ├── engagement_indices/
│   ├── statistical_analysis/
│   └── results/
├── fMRI_Analysis/
│   ├── preprocessing/
│   ├── ICA_components/
│   ├── RSN_classification/
│   └── results/
├── papers/
│   ├── Group5_EEGLab_paper.pdf
│   └── Group5_FSL_paper.pdf
└── README.md
Key Concepts
EEG Frequency Bands

Delta (δ): 0.5-4 Hz
Theta (θ): 4-8 Hz
Alpha (α): 8-12 Hz
SMR: 12-15 Hz
Beta (β): 15-35 Hz
Gamma (γ): 35-70 Hz

Circumplex Model of Affect
Emotions positioned in 2D space:

Valence: Pleasant ↔ Unpleasant
Arousal: Activation ↔ Deactivation

Analyzed emotion pairs:

Happy ↔ Sad
Pleasant (Relaxed) ↔ Afraid
Tender (Serene) ↔ Angry
Energetic (Excited) ↔ Tense (Alarmed)

Clinical Implications
Both studies contribute to understanding brain function with potential applications in:

Mental health diagnostics (depression, schizophrenia)
Emotion recognition systems
Music therapy interventions
Brain-computer interfaces
Cognitive neuroscience research

Limitations and Future Directions
EEG Study

Individual variability in emotional responses to music
Subjectivity of self-reported emotional states
Inherent noisiness of EEG signals
Future work: Integration with physiological markers (heart rate, skin conductance) and behavioral indicators

fMRI Study

Limited cerebellar coverage due to field-of-view constraints
Manual classification is time-consuming
Future work: Combine manual classification with automated denoising (FIX, ICA-AROMA) for standardization

Authors

Giuseppe Legista - Università Politecnica delle Marche (UNIVPM)
Mauro Silveri - Università Politecnica delle Marche (UNIVPM)
Davide Pitucci - Università Politecnica delle Marche (UNIVPM)
Christian Di Salvo - Università Politecnica delle Marche (UNIVPM)

References
EEG Study

Liu et al. (2021) - Review on emotion recognition based on EEG
Marcantoni et al. (2023) - Ratio indexes for mental involvement assessment
Pope et al. (1995) - Biocybernetic system for operator engagement
Nasuto et al. (2020) - EEG dataset during affective music listening
Russell (1980) - Circumplex Model of Affect

fMRI Study

Smith et al. (2009) - Brain functional architecture during activation and rest
Griffanti et al. (2017) - Hand classification of fMRI ICA noise components
Salimi-Khorshidi et al. (2014) - Automatic denoising of fMRI data
Lee et al. (2013) - Resting-state fMRI methods and clinical applications

License
This project is part of academic research at Università Politecnica delle Marche.

For questions or collaborations, please contact the authors at their respective university email addresses.
