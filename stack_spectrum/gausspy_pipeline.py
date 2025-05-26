import numpy as np
import pickle
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class GausspyPipeline:
    """
    A pipeline class for automatic spectral Gaussian decomposition and stacking,
    using GaussPy. Supports raw stacking, peak alignment stacking, frequency alignment,
    and double-shifting of already stacked spectra.
    """

    def __init__(
        self,
        cube_file,
        v1, v2, x1, x2, y1, y2,
        input_pickle='spectrum_for_gausspy.pickle',
        result_pickle='gausspy_result.pickle',
        alpha1=0.1,
        alpha2=12.0,
        snr_thresh=3.0,
        plot_max=10,
        plot_dpi=100,
        stack_vrange=(-200, 200),
        stack_dv=0.2,
        normalize=False,
        min_valid_spectra=30
    ):
        self.cube_file = cube_file
        self.v1, self.v2 = v1, v2
        self.x1, self.x2 = x1, x2
        self.y1, self.y2 = y1, y2
        self.input_pickle = input_pickle
        self.result_pickle = result_pickle
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.snr_thresh = snr_thresh
        self.plot_max = plot_max
        self.plot_dpi = plot_dpi
        self.stack_vrange = stack_vrange
        self.stack_dv = stack_dv
        self.normalize = normalize
        self.min_valid_spectra = min_valid_spectra

        # 自動擷取 RESTFREQ（如 header 沒有則為 None）
        header = fits.getheader(cube_file)
        self.restfreq = header.get('RESTFREQ', None)

    def prepare_input(self):
        """
        Extract spectra from a FITS cube (spatial subregion) and save as a pickle
        for GaussPy input.
        """
        data = fits.getdata(self.cube_file)[0]
        header = fits.getheader(self.cube_file)
        subcube = data[self.v1:self.v2, self.y1:self.y2, self.x1:self.x2]

        CRVAL3 = header['CRVAL3']
        CDELT3 = header['CDELT3']
        CRPIX3 = header['CRPIX3']
        N_full = header['NAXIS3']

        chan_all = np.arange(N_full)
        velo_all = CRVAL3 + (chan_all + 1 - CRPIX3) * CDELT3
        velo_all_kms = velo_all / 1e3
        velo = velo_all_kms[self.v1:self.v2]

        ny, nx = subcube.shape[1:]
        spectra_list = []
        errors_list = []

        for j in range(ny):
            for i in range(nx):
                spec = subcube[:, j, i]
                rms = np.nanstd(spec)
                spectra_list.append(spec)
                errors_list.append(np.full_like(spec, rms))

        output = {
            'x_values': [velo] * len(spectra_list),
            'data_list': spectra_list,
            'errors': errors_list
        }

        with open(self.input_pickle, 'wb') as f:
            pickle.dump(output, f)

        print(f"Prepared {len(spectra_list)} spectra → {self.input_pickle}")

    def run_decomposition(self):
        """
        Runs GaussPy phase-two decomposition using the pre-saved spectra pickle.
        """
        from gausspy.gp import GaussianDecomposer
        g = GaussianDecomposer()
        g.set('phase', 'two')
        g.set('alpha1', self.alpha1)
        g.set('alpha2', self.alpha2)
        g.set('SNR_thresh', [self.snr_thresh, self.snr_thresh])

        print("Running GaussPy phase-two decomposition ...")
        result = g.batch_decomposition(self.input_pickle)

        with open(self.result_pickle, 'wb') as f:
            pickle.dump(result, f)

        print(f"GaussPy decomposition finished → {self.result_pickle}")
        print(f"Result keys: {list(result.keys())}")

    def count_fits(self):
        """
        Counts how many fits are successful. Optionally returns failed pixel indices.
        """
        with open(self.result_pickle, 'rb') as f:
            result = pickle.load(f)

        amps_all = result.get('amplitudes_fit')
        if amps_all is None:
            raise KeyError("Missing 'amplitudes_fit' in results.")

        n_total = len(amps_all)
        failed_indices = [i for i, amps in enumerate(amps_all) if amps is None or len(amps) == 0]
        n_success = n_total - len(failed_indices)
        success_rate = 100 * n_success / n_total if n_total > 0 else 0

        print(f"Total: {n_total}, Success: {n_success}, Failed: {len(failed_indices)}")
        print(f"Success rate: {success_rate:.1f}%")
        return failed_indices

    def stack_raw_spectra(self, plot=True):
        """
        Stack (average) all raw spectra without any shifting.
        Returns velocity axis, mean spectrum, and std spectrum.
        """
        with open(self.input_pickle, 'rb') as f:
            data = pickle.load(f)
        x_all = data['x_values']
        y_all = data['data_list']
        v_axis = x_all[0]
        y_matrix = np.array(y_all)
        mean_spec = np.nanmean(y_matrix, axis=0)
        std_spec = np.nanstd(y_matrix, axis=0)
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(v_axis, mean_spec, label='Raw Stacked Spectrum', color='black')
            # plt.fill_between(v_axis, mean_spec-std_spec, mean_spec+std_spec, color='gray', alpha=0.3, label='±1σ')
            plt.xlabel('Velocity (km/s)')
            plt.ylabel('Averaged Intensity')
            plt.title('Raw Stacked Spectrum (No Shifting)')
            plt.grid(True, linestyle=':')
            plt.legend()
            plt.tight_layout()
            plt.show()
        print(f"Stacked {len(y_all)} raw spectra.")
        return v_axis, mean_spec, std_spec

    def stack_restframe(self, plot=True):
        """
        Stack all successful spectra by aligning the main component to v=0 km/s.
        Returns (velocity grid, mean spectrum, std spectrum).
        """
        import pickle
        from scipy.interpolate import interp1d
        import numpy as np
        import matplotlib.pyplot as plt

        with open(self.input_pickle, 'rb') as f:
            data = pickle.load(f)
        with open(self.result_pickle, 'rb') as f:
            result = pickle.load(f)

        x_all = data['x_values']
        y_all = data['data_list']
        amps_all = result['amplitudes_fit']
        means_all = result['means_fit']

        v_grid = np.arange(
            self.stack_vrange[0], self.stack_vrange[1] + self.stack_dv, self.stack_dv
        )
        stacked = []
        used_indices = []

        for i, (x, y, amps, means) in enumerate(zip(x_all, y_all, amps_all, means_all)):
            if amps is None or len(amps) == 0 or means is None:
                continue

            idx_peak = np.argmax(amps)
            v_peak = means[idx_peak]
            x_shifted = x - v_peak

            try:
                f_interp = interp1d(x_shifted, y, kind='linear', bounds_error=False, fill_value=np.nan)
                y_interp = f_interp(v_grid)
                if self.normalize:
                    y_interp = y_interp / np.nanmax(np.abs(y_interp))
                stacked.append(y_interp)
                used_indices.append(i)
            except Exception:
                continue

        stacked = np.array(stacked)
        mean_spec = np.nanmean(stacked, axis=0)
        std_spec = np.nanstd(stacked, axis=0)

        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(v_grid, mean_spec, label='Stacked Spectrum', color='black')
            plt.axvline(0, linestyle='--', color='red', label='v = 0 km/s')
            plt.xlabel('Rest-frame Velocity (km/s)')
            plt.ylabel('Stacked Intensity')
            plt.title('Rest-frame Stacked Spectrum')
            plt.grid(True, linestyle=':')
            plt.legend()
            plt.tight_layout()
            plt.show()

        n_total = len(x_all)
        n_used = len(used_indices)
        print(f'Stacking done: {n_used}/{n_total} spectra used ({100 * n_used / n_total:.1f}%)')

        return v_grid, mean_spec, std_spec


    def plot_fits(self):
        """
        Plot spectra and their fitted Gaussian components.
        """
        with open(self.input_pickle, 'rb') as f:
            data = pickle.load(f)
        with open(self.result_pickle, 'rb') as f:
            result = pickle.load(f)

        x_all = data['x_values']
        y_all = data['data_list']
        amps_all = result['amplitudes_fit']
        means_all = result['means_fit']
        fwhms_all = result['fwhms_fit']

        n_result = len(amps_all)
        n_data = len(y_all)
        n_total = min(n_result, n_data)
        n_plotted = 0

        for i in range(n_total):
            if self.plot_max is not None and n_plotted >= self.plot_max:
                break

            amps = amps_all[i]
            means = means_all[i]
            fwhms = fwhms_all[i]

            if amps is None or len(amps) == 0:
                continue

            x = x_all[i]
            y = y_all[i]
            stddevs = fwhms / (2 * np.sqrt(2 * np.log(2)))

            fit_total = np.zeros_like(x)
            gaussians = []

            for amp, mean, std in zip(amps, means, stddevs):
                gauss = amp * np.exp(-(x - mean)**2 / (2 * std**2))
                fit_total += gauss
                gaussians.append((gauss, mean, amp))

            plt.figure(figsize=(10, 4), dpi=self.plot_dpi)
            plt.plot(x, y, color='black', lw=1, label='Spectrum')
            plt.plot(x, fit_total, color='red', lw=2, label='Total Fit')

            colors = plt.cm.tab10(np.linspace(0, 1, len(gaussians)))

            for j, (g, mean, amp) in enumerate(gaussians):
                plt.plot(x, g, linestyle='--', color=colors[j % 10], alpha=0.8, label=f'Comp {j+1}')
                plt.axvline(mean, color=colors[j % 10], linestyle='-', linewidth=1)
                plt.text(mean, amp * 1.05, f'v = {mean:.1f} km/s',
                         rotation=90, ha='center', va='bottom', fontsize=8,
                         color=colors[j % 10], backgroundcolor='white')

            plt.title(f'Gaussian Fit - Spectrum #{i}')
            plt.xlabel('Velocity (km/s)')
            plt.ylabel('Intensity')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(fontsize=8, loc='upper right')
            plt.tight_layout()
            plt.show()

            n_plotted += 1

        print(f"Plotted {n_plotted} fitted spectra.")

    def config(self):
        """
        Return current pipeline parameter settings as a dict.
        """
        return self.__dict__

    