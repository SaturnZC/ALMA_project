# Introduction
This is using `gausspy` package to do the Gaussian fit, find the peak and move to the rest frame. Stack the spectrum to enhance the SNR to make the relatively weak line visible.
# Instruction
Ensure that all necessary dependencies are installed. Before installation, use Anaconda3 to manage the development environment. You can type:
    '''
    conda create --name astroimgAna python=3.7        
    pip install --upgrade pip      
    pip install numpy scipy pandas matplotlib   
    pip install astropy==4.3.1   
    pip install jupyter   
    pip install h5py   
    pip install PyAVM healpy
    '''
    # Log
- 2025/5/29 
    - upgrade gausspy pipeline
	- add pixel location to pickle
	- there is a GaussPy batch bug, we can't decomposition serval spctrum together on my windows pc, haven't try run it on server yet.
	- 