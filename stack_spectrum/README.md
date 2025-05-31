# Introduction  

This project uses the `gausspy` package to perform Gaussian fitting, identify peaks, and shift to the rest frame. The spectrum is stacked to enhance the SNR, making relatively weak lines visible.
  
---
# Instruction  
Ensure that all necessary dependencies are installed. For easier installation, a `environment.yml` file is provided. Before installation, download and install [Anaconda3](https://www.anaconda.com/products/distribution) to manage the development environment. You can type:

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
---
# Pipeline
`gausspy_pipeline.py` is the main script of all the function. We can import it like:  

```python
from gausspy_pipeline import GausspyPipeline
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle
from astropy.io import fits

pipeline = GausspyPipeline(
    cube_file='../datacubes/spw0.fits', #your fits file
    v1=10, v2=1500, x1=190, x2=210, y1=220, y2=240, # v : frequency:cahnnel, (x,y) : pixel index 
    alpha1=0.1, alpha2=12.0, snr_thresh=3.0, #alpha: train parameter
    stack_vrange=(-200, 200), stack_dv=0.2
)

```

---
# Log  
  
- 2025-05-31
	- add a dic to sort the single peak , multiple peak, failed peak
	- add a function `plot_stacked_zoom` to plot the compare of stack rest frame and raw more easier.  
- 2025-05-29
	- upgrade gausspy pipeline
	- add pixel location to pickle
	- there is a GaussPy batch bug, we can't decomposition serval spctrum together on my windows pc, haven't tried running it on a server yet.
	  As a temporary workaround, consider running the decomposition on a Linux server or using smaller batches of spectra. Further debugging is required to identify the root cause.