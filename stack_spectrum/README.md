# Introduction  

This project uses the `gausspy` package to perform Gaussian fitting, identify peaks, and shift to the rest frame. The spectrum is stacked to enhance the SNR, making relatively weak lines visible.
  
---
# Instruction  
Ensure that all necessary dependencies are installed. For easier installation, a `environment.yml` file is provided. Before installation, use Anaconda3 to manage the development environment. You can type:

```bash
conda env create -f environment.yml
conda activate gausspy
pip install --upgrade pip
pip install numpy scipy pandas matplotlib
pip install astropy==4.3.1
pip install jupyter
pip install h5py
pip install PyAVM healpy
```

# Log  
  
- 2025/5/29 
    - upgrade gausspy pipeline
	- add pixel location to pickle
	- there is a GaussPy batch bug, we can't decomposition serval spctrum together on my windows pc, haven't tried running it on a server yet.
	  As a temporary workaround, consider running the decomposition on a Linux server or using smaller batches of spectra. Further debugging is required to identify the root cause.