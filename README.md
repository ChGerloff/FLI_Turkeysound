# FLI_Turkeysound

This code was initially made for the detection of agonistic(cannibalistic) pecking occurring in commercially housed turkey birds. Sadly this did not work out since either the data from the microphones did not save those days, or the cameras used for validating the pecking reset and instead recorded the corner of the barn, therefore being unusable to validate at which time the pecking occurred. 

By chance a digestive sickness broke out in one of the 12 compartments. It was not found out what specific illness occurred but after treatment it was cured. This project was then readjusted to create a Neural Network to detect the beginning of the sickness based on the sounds the turkeys made at that time. 

The code is divided into three parts. The data is gathered by microphones suspended from the ceiling of 12 compartments within a barn at the Friedrich-Loeffler Institute for animal walfare and husbandry in Celle. The initial data is saved in .wav files with a length of 55 seconds. 

1. Data Augmentation

In the first step, the .wav files are converted into spectrograms, making them more readable for humans and more usable for the Neural Networks. Additionally the spectrograms are split into 4 parts and an overlap is incorporated. The overlap creates spectrograms in which the last part of the first split-spectrogram is the same as the first seconds of the second spectrogram. This is done to imitate the rotation normally done in image analysis since spectrograms can't be rotated without becoming unusable. 

2. Convolutional Neural Network for the data labeled sick or healthy

The data was labeled into sick and healthy data. The sick labeled data are the sound files of the days in which we could be sure that the sickness was present and the healthy labeled data were the days before and after this time window, for which we could be sure that the sickness was not yet present or the treatment was finished.

3. Neural Network for individual days

Additionally to the first analysis a second analysis was done in which each of the individual days of two weeks which include the days in which the sickness was present in the compartment. This is done since the neural network in theory should have an easier time to accurately detect the sick days. Additionally when analysing the confusion matrix the days falsely categorized as another day should be categorized also as a sick labeled day if the correct label was sick and vise versa with the healthy days.  

[Confusion_4split.pdf](https://github.com/user-attachments/files/32165117/Confusion_4split.pdf)
