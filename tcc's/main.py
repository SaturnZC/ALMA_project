import os, sys, time

sys.path
sys.path.append('./')

import numpy as np
from scipy.interpolate import interp1d
from astropy import wcs
from astropy.io import fits

import matplotlib
import matplotlib.pyplot as plt

import gausspy.gp as gp
import pickle


# Global
c = 299792458 # the speed of light

# Common used function
def plot_spectra( xvalue_array    = np.array([]), 
                  intensity_array = np.array([]),
                  figsize = (6,4), 
                  xlabel = 'Frequency [GHz]'    , xlabel_size = 12, xscale = 'linear',
                  ylabel = 'Intensity [Jy/beam]', ylabel_size = 12, yscale = 'linear',
                  xmin = None, xmax = None, ymin = None, ymax = None,
                  datalabel = 'TBD', linewidth = 2.0, 
                  color     = (0.7,0.7,0.2, 0.3),
                  fontsize  = 12, 
                  plot_ticks = True, xtick_size = 12, ytick_size = 12,
                  gaussians = None, text = None, 
                  textcolor = 'grey', textsize = 30, 
                  text_x_loc = 0.08, text_y_loc = 0.8,
                  outfile = 'none'
                ):
    '''
    This is a function to plot 1D spectra.
    '''

    # Initializing figure
    fig = plt.figure(figsize = (figsize[0], figsize[1]))
    ax = fig.add_axes([0.12, 0.1, 0.75, 0.75])
        
    # Set the x/y axis title and legend
    ax.set_xlabel(xlabel, size = xlabel_size)
    ax.set_ylabel(ylabel, size = ylabel_size)

    # set plot scale
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    # set the x,y limit 
    if xmin is not None and xmax is not None:
        ax.xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        ax.ylim(ymin, ymax)

    # set label fontsizes
    ax.tick_params(axis='x', labelsize=xtick_size)
    ax.tick_params(axis='y', labelsize=ytick_size)
    if not plot_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        
    #plt.ticklabel_format(axis='x', style='plain')
        
    # plot the data
    if datalabel == 'none':
        ax.plot(xvalue_array, intensity_array,
                 '-', # symbol shape
                 color=color, # (R, G, B, transparency), ranged between [0, 1]
                 linewidth = linewidth
                )
    else:
        ax.plot(xvalue_array, intensity_array,
                 '-', # symbol shape
                 color = color, # (R, G, B, transparency), ranged between [0, 1]
                 linewidth = linewidth, 
                 label = datalabel
                )  
        # Setting the figure legend 
        ax.legend(loc = 1, fontsize = fontsize)
        
    # plot the gaussian if needed
    if gaussians is not None:
        for amp, fwhm, mean in gaussians:
            yy = amp * np.exp(-4. * np.log(2) * (xvalue_array - mean)**2 / fwhm**2)
            ax.plot(xvalue_array, yy, '-', lw=1.5, color='red', label='Gaussian Component')

    # Add index label if provided
    if text is not None:
        ax.text(text_x_loc, text_y_loc, f'{text}', 
                 verticalalignment='bottom', horizontalalignment='left',
                 color = textcolor, transform=ax.transAxes, fontsize = textsize)

    if outfile != 'none':
        plt.savefig(outfile, transparent=False, bbox_inches='tight')
        plt.close(fig)

    plt.show()

# functions for fitting with gaussp
def Gtransdata(cube, velo, freq, rms = 0.008, base='velo',
               xrange1 = 0, xrange2 = None, 
               yrange1 = 0, yrange2 = None,
               spw_id = '0'):
    '''
    Transform FITS datacube to GaussPy format.
    '''

    # Specify necessary parameters
    FILENAME_DATA_GAUSSPY = 'cube_' + spw_id + '.pickle'

    ## initialize gausspy set
    data   = {} #dict
    errors = np.ones(cube.shape[1]) * rms
    
    if base == 'velo':
        chan = velo / 1e3
    elif base == 'freq':
        chan = freq / 1e9
    else :
        chan = np.arange(cube.shape[1])

    # cycle through each spectrum
    xlength = cube.shape[3]
    ylength = cube.shape[2]
    
    if xrange2 is None:
        xrange2 = xlength
    if yrange2 is None:
        yrange2 = ylength
        
        
    for xpix in range(xrange1, xrange2):
        for ypix in range(yrange1, yrange2):

            # get the spectrum
            intensity = cube[0][:, ypix, xpix]

            # get the spectrum location
            location = (xpix, ypix)
    
            # Enter results into GaussPy-friendly dataset
            data['data_list'] = data.get('data_list', []) + [intensity]
            data['x_values']  = data.get('x_values', [])  + [chan]
            data['errors']    = data.get('errors', [])    + [errors]
            data['location']  = data.get('location', [])  + [location]

    # Save decomposition information
    pickle.dump(data, open(FILENAME_DATA_GAUSSPY, 'wb'))

def Gdecompose(alpha1 = 0.1, alpha2 = 12.0, snr_thresh = 3.0, spw_id = '0'):
    '''
    Decompose multiple Gaussian dataset using AGD with TRAINED alpha.
    '''

    # Specify necessary parameters
    alpha1 = alpha1
    alpha2 = alpha2
    snr_thresh = snr_thresh

    FILENAME_DATA_GAUSSPY = 'cube_' + spw_id +'.pickle'
    FILENAME_DATA_DECOMP = 'cube_decomposed_' + spw_id + '.pickle'

    # Load GaussPy
    g = gp.GaussianDecomposer()

    # Setting AGD parameters
    g.set('phase', 'two')
    g.set('SNR_thresh', [snr_thresh, snr_thresh])
    g.set('alpha1', alpha1)
    g.set('alpha2', alpha2)

    # Run GaussPy
    decomposed_data = g.batch_decomposition(FILENAME_DATA_GAUSSPY)

    # Save decomposition information
    pickle.dump(decomposed_data, open(FILENAME_DATA_DECOMP, 'wb'))

# Major flow of data analysis
class spectraAna:
    '''
    Class for analyzing image cube. The main purpose of this analysis is to
    stack the un-redshifted/blueshifted spectra.
    '''
    
    ### Constructor and destructor
    def __init__(self, fitscubename = ' '):
        self.fitscubename = fitscubename
    
    def __del__(self):
        pass
    
    ### Methods in this class
    def readfits(self, verbose = False):
        if verbose == True:
            print('This is the program to read FITS image cube')
            
        if self.fitscubename != ' ':
            if verbose == True:
                print('Reading FITS image cube :' + self.fitscubename)
        
        # import FITS image to HDU
        self.Ihdu = fits.open(self.fitscubename)
        
        # choose the certain information
        self.header = self.Ihdu[0].header
        self.naxis1 = self.Ihdu[0].header['naxis1']
        self.naxis2 = self.Ihdu[0].header['naxis2']
        self.naxis3 = self.Ihdu[0].header['naxis3']
        self.crpix3 = self.Ihdu[0].header['crpix3']
        self.cdelt3 = self.Ihdu[0].header['cdelt3']
        self.crval3 = self.Ihdu[0].header['crval3']
        self.ctype3 = self.Ihdu[0].header['ctype3']
        self.restfreq = self.Ihdu[0].header['RESTFREQ']
        self.cube = self.Ihdu[0].data # 4d array
        
        # create the frequency array
        if self.ctype3 == 'VELO-LSR':
            self.velo_array = self.crval3 + ( np.arange(self.naxis3) + 1 - self.crpix3 ) * self.cdelt3
            self.freq_array = self.restfreq * (1 - self.velo_array / c)

        elif self.ctype3 == 'FREQ':
            self.freq_array = self.crval3 + ( np.arange(self.naxis3) + 1 - self.crpix3 ) * self.cdelt3
            self.velo_array = c * (1 - (self.freq_array / self.restfreq) )
        
    def fit_1dgauss(self, verbose = False, 
                    cube = None, velo = np.array([]), freq = np.array([]), 
                    rms  = 0.08  , base = 'velo', spw_id = '0',
                    xrange1 = 0, xrange2 = None,
                    yrange1 = 0, yrange2 = None,
                   ):
        '''
        The task to fit centroid frequency pixel-by-pixel.
        Noticed that the input data(cube) need to be 4 dimensional array.
        '''

        Gtransdata( cube = cube, velo = velo,  freq = freq,
                    spw_id = spw_id, rms = rms, base = 'velo',
                    xrange1 = xrange1, xrange2 = xrange2,
                    yrange1 = yrange1, yrange2 = yrange2 )
        
        Gdecompose(spw_id = spw_id)

    def unredshift(self, freq_array = np.array([]), velo_array = np.array([]),
                   spw_restfreq = 215.700000E+9,
                   fit_linefreq = 215.219259E+9,
                   base = 'velo', centroid_value= 0 ):
        '''
        This function is to un-redshift/blueshift spectra.
        '''

        # use frequency array to remove Doppler shift
        if base == 'freq':
            delta_freq = centroid_value * 1e9 - fit_linefreq 
            shift_freq = freq_array - delta_freq
      
            return shift_freq, delta_freq
        
        # use velocity array to remove Doppler shift
        elif base == 'velo':
            fit_restvelo  = c * (1 - fit_linefreq / spw_restfreq)
            delta_velo = centroid_value * 1e3 - fit_restvelo 
            shift_velo = velo_array - delta_velo
                        
            return shift_velo, delta_velo

    def stack_spectra(self, newx_array, shifted_x_list, intensity_list,
                      option = 'unredshift' ):
        '''
        The task to stack the unredshifted/unblueshifted spectra and plot the stacked spectra
        '''

        # create the stacked intensity array with unredshift
        if option == 'unredshift':
            sum_sm_intensity = np.zeros_like(newx_array)

            # stack
            for x, inten in zip(shifted_x_list, intensity_list):
                interp_func = interp1d(x, inten, kind='linear', bounds_error=False, fill_value=0)
                sum_sm_intensity += interp_func(newx_array)

            average_intensity = sum_sm_intensity / len(intensity_list)

            return average_intensity
                
        # create the stacked intensity array without unredshift
        elif option == 'redshift':
            average_intensity = []

            for i in range(0, self.naxis3):
                temp = 0

                for inten in intensity_list:
                    temp += inten[i] / len(intensity_list)

                average_intensity.append(temp)

            return average_intensity

