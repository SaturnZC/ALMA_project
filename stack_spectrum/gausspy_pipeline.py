import numpy as np
import pickle
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class GausspyPipeline:
    """
    A pipeline class for automatic spectral Gaussian decomposition and stacking,
    using GaussPy. All parameters are set during initialization, and all steps
    are accessible as methods, with no need to repeatedly specify parameters.
    """

class GausspyPipeline:
    def __init__(
        self,
        cube_file,
        v1, v2, x1, x2, y1, y2,
        input_pickle='spectrum_for_gausspy.pickle',
        result_pickle='gausspy_result.pickle',
        alpha1=2.0,
        alpha2=6.0,
        snr_thresh=3.0,
        plot_max=10,
        plot_dpi=100,
        stack_vrange=(-100, 100),
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

    def prepare_input(self):
        """
        Extracts spectra from a FITS cube (spatial subregion) and saves as a pickle
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

    def plot_fits_restframe(self):
        """
        Plot spectra with main Gaussian component aligned at v=0 km/s (rest frame).
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

            x_orig = x_all[i]
            y = y_all[i]

            v_peak = means[0]
            x_shifted = x_orig - v_peak
            means_shifted = [v - v_peak for v in means]

            stddevs = fwhms / (2 * np.sqrt(2 * np.log(2)))

            fit_total = np.zeros_like(x_orig)
            gaussians = []

            for amp, mean, std in zip(amps, means_shifted, stddevs):
                gauss = amp * np.exp(-(x_shifted - mean)**2 / (2 * std**2))
                fit_total += gauss
                gaussians.append((gauss, mean, amp))

            plt.figure(figsize=(10, 4), dpi=self.plot_dpi)
            plt.plot(x_shifted, y, color='black', lw=1, label='Spectrum')
            plt.plot(x_shifted, fit_total, color='red', lw=2, label='Total Fit')

            colors = plt.cm.tab10(np.linspace(0, 1, len(gaussians)))

            for j, (g, mean, amp) in enumerate(gaussians):
                plt.plot(x_shifted, g, linestyle='--', color=colors[j % 10], alpha=0.8, label=f'Comp {j+1}')
                plt.axvline(mean, color=colors[j % 10], linestyle='-', linewidth=1)
                plt.text(mean, amp * 1.05, f'{mean:.1f} km/s',
                         rotation=90, ha='center', va='bottom', fontsize=8,
                         color=colors[j % 10], backgroundcolor='white')

            plt.title(f'Gaussian Fit (Rest Frame) - Spectrum #{i}')
            plt.xlabel('Rest-frame Velocity (km/s)')
            plt.ylabel('Intensity')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(fontsize=8, loc='upper right')
            plt.tight_layout()
            plt.show()

            n_plotted += 1

        print(f"Plotted {n_plotted} rest-frame aligned spectra.")

    def stack_restframe(self):
        """
        Stack all successful spectra by aligning the main component to v=0 km/s.
        Returns (velocity grid, mean spectrum, std spectrum).
        """
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

        plt.figure(figsize=(10, 4))
        plt.plot(v_grid, mean_spec, label='Stacked Spectrum', color='black')
        plt.fill_between(v_grid, mean_spec - std_spec, mean_spec + std_spec, color='gray', alpha=0.3, label='±1σ')
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

    def compare_stack(self):
        """
        Compare stacking with and without rest-frame alignment.
        """
        with open(self.input_pickle, 'rb') as f:
            data = pickle.load(f)
        with open(self.result_pickle, 'rb') as f:
            result = pickle.load(f)

        x_all = data['x_values']
        y_all = data['data_list']
        amps_all = result['amplitudes_fit']
        means_all = result['means_fit']

        v_grid = np.arange(self.stack_vrange[0], self.stack_vrange[1] + self.stack_dv, self.stack_dv)
        stacked_shifted, stacked_direct = [], []
        used_count = 0

        for x, y, amps, means in zip(x_all, y_all, amps_all, means_all):
            if amps is None or means is None or len(amps) == 0:
                continue

            try:
                idx_peak = np.argmax(amps)
                v_peak = means[idx_peak]

                x_shifted = x - v_peak
                interp_shifted = interp1d(x_shifted, y, kind='linear', bounds_error=False, fill_value=0)
                y_shifted = interp_shifted(v_grid)
                if self.normalize:
                    max_val = np.nanmax(np.abs(y_shifted))
                    if max_val > 0:
                        y_shifted /= max_val
                stacked_shifted.append(y_shifted)

                interp_direct = interp1d(x, y, kind='linear', bounds_error=False, fill_value=0)
                y_direct = interp_direct(v_grid)
                if self.normalize:
                    max_val = np.nanmax(np.abs(y_direct))
                    if max_val > 0:
                        y_direct /= max_val
                stacked_direct.append(y_direct)

                used_count += 1
            except Exception:
                continue

        stacked_shifted = np.array(stacked_shifted)
        stacked_direct = np.array(stacked_direct)

        valid_counts = np.sum(~np.isnan(stacked_shifted), axis=0)
        valid_mask = valid_counts >= self.min_valid_spectra
        v_grid_valid = v_grid[valid_mask]

        mean_shifted = np.nanmean(stacked_shifted[:, valid_mask], axis=0)
        std_shifted = np.nanstd(stacked_shifted[:, valid_mask], axis=0)
        mean_direct = np.nanmean(stacked_direct[:, valid_mask], axis=0)
        std_direct = np.nanstd(stacked_direct[:, valid_mask], axis=0)

        plt.figure(figsize=(10, 5))
        plt.plot(v_grid_valid, mean_direct, label='Original (no alignment)', color='gray', linestyle='-')
        plt.fill_between(v_grid_valid, mean_direct - std_direct, mean_direct + std_direct, color='gray', alpha=0.2)

        plt.plot(v_grid_valid, mean_shifted, label='Rest-frame aligned', color='black')
        plt.fill_between(v_grid_valid, mean_shifted - std_shifted, mean_shifted + std_shifted, color='black', alpha=0.2)

        plt.axvline(0, linestyle='--', color='red', label='v = 0 km/s')
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Intensity')
        plt.title('Stacked Spectrum Comparison')
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"Stacking comparison done. Used {used_count} spectra.")

    def config(self):
        """
        Return current pipeline parameter settings as a dict.
        """
        return self.__dict__
