# MultiStage-Grassmannian-DNN
Code of "Recursive CSI Quantization of Time-Correlated MIMO Channels by Deep Learning Classification", IEEE SPL 2020

Contact: Stefan Schwarz, Institute of Telecommunications, TU Wien, stefan.schwarz@tuwien.ac.at

This code can be used to reproduce the neural network quantization results of 

"Recursive CSI Quantization of Time-Correlated MIMO Channels by Deep Learning Classification", S. Schwarz, IEEE SPL, 2020

The code is setup for a small-scale MIMO system with 4 transmit and 2 receive antennas, in order to speed-up the execution. However, these parameters
can be changed in "Quant_example.m". 

The code requires Matlab's Deep Learning Toolbox.

To rund the code, exectue the main file "Quant_example.m".

This file will call the scripts "NN_training.m" and "train_net.me" for DNN training.

Afterwards, "time_corr.m" will be executed to apply the trained multi-stage quantizer for quantization of time-correlated channels.
