## Dataset Description
Each set of raw data is obtained at a constant nominal rotational speed and load. Four fault types are collected: inner race, outer race, ball, and cage (five of each type). A total of 20 bearings are tested, where each bearing has three states of data collection: healthy, developing fault, and faulty. As a result, 60 distinct sets of data are included. For all cases, data is collected at a sampling rate of 42,000 Hz for a total 10 seconds per set of data. Raw data is provided in the folders as comma separated values files (.csv), Excel files (.xlsx), and MATLAB files (.mat)

Spectrograms are created from the raw data by using the short-time Fourier transform (STFT), while using a Hanning window to convert the accelerometer and microphone values into 2D images. For each raw data file, 400 spectrogram images are created with a signal length of 512. A total of 24,000 spectrogram images are created, 12,000 images for each of the accelerometer and microphone data files. The processed data are provided in the zip files as portable network graphics files (.png).


The dataset used in this project can be found here https://data.mendeley.com/datasets/y2px5tg92h/5
