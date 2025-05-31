This is using `gausspy` package to do the Gaussian fit, find the peak and move to the rest frame. Stack the spectrum to improve the SNR.

- 2025/5/29 
    - upgrade gausspy pipeline
	- add pixel location to pickle
	- there is a GaussPy batch bug, we can't decomposition serval spctrum together on my windows pc, haven't try run it on server yet.
	- 