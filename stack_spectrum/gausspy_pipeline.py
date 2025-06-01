import re
import os
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
            input_pickle=None,
            result_pickle=None,
            alpha1=0.1,
            alpha2=12.0,
            snr_thresh=3.0,
            plot_max=100,
            plot_dpi=100,
            stack_vrange=(-200, 2000),
            stack_dv=0.2,
            normalize=False,
            min_valid_spectra=30
        ):
            self.cube_file = cube_file
            self.v1, self.v2 = v1, v2
            self.x1, self.x2 = x1, x2
            self.y1, self.y2 = y1, y2

            # ==== 自動判斷 spw 編號（可支援路徑內 spwN 或 spw_N）====
            basename = os.path.basename(cube_file)
            m = re.search(r'spw[_]?(\d+)', basename)
            spw_tag = f"spw{m.group(1)}" if m else "spw"

            # ==== input/result pickle 根據 spw 編號自動命名（除非手動覆蓋）====
            self.input_pickle = input_pickle or f"spectrum_for_gausspy_{spw_tag}.pickle"
            self.result_pickle = result_pickle or f"gausspy_result_{spw_tag}.pickle"

            self.alpha1 = alpha1
            self.alpha2 = alpha2
            self.snr_thresh = snr_thresh
            self.plot_max = plot_max
            self.plot_dpi = plot_dpi
            self.stack_vrange = stack_vrange
            self.stack_dv = stack_dv
            self.normalize = normalize
            self.min_valid_spectra = min_valid_spectra

            # 自動擷取 RESTFREQ
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
        locations = []

        for j in range(ny):
            for i in range(nx):
                spec = subcube[:, j, i]
                rms = np.nanstd(spec)
                spectra_list.append(spec)
                errors_list.append(np.full_like(spec, rms))
                locations.append((self.y1 + j, self.x1 + i))  # 儲存對應的 pixel 全域座標

        output = {
            'x_values': [velo] * len(spectra_list),
            'data_list': spectra_list,
            'errors': errors_list,
            'location': locations 
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
    
    
    def classify_fit_results(self, output_pickle='fit_result_dic.pickle', exclude_range=(-3000, 3000)):
        """
        分類所有 pixel 的高斯擬合結果為單峰、多峰、失敗
        並儲存 index, location, amps, means, sigmas
        只要有任何一個分量中心位置超過 exclude_range，就直接踢掉這條光譜。
        exclude_range: (vmin, vmax)，允許的分量中心範圍（單位: km/s）
        """
        with open(self.input_pickle, 'rb') as f:
            data = pickle.load(f)
        with open(self.result_pickle, 'rb') as f:
            result = pickle.load(f)

        fit_result_dic = {'s':[], 'm':[], 'f':[]}
        fwhm2sigma = lambda fwhm: np.array(fwhm) / 2.35482 if fwhm is not None else None

        n_skipped = 0  # 記錄被踢掉的數量
        vmin, vmax = exclude_range

        for i, (loc, amps, means, fwhms) in enumerate(zip(
            data['location'],
            result['amplitudes_fit'],
            result['means_fit'],
            result['fwhms_fit'])):

            loc = tuple(loc)
            peak_num = len(means) if means is not None else 0
            sigmas = fwhm2sigma(fwhms) if fwhms is not None else None

            # 只要任何一個分量中心超過 exclude_range，就踢掉
            if means is not None and len(means) > 0:
                means_arr = np.array(means)
                if np.any((means_arr < vmin) | (means_arr > vmax)):
                    n_skipped += 1
                    continue

            pixel_dict = {
                "index": i,
                "location": loc,
                "amps": amps,
                "means": means,
                "sigmas": sigmas,
            }

            if peak_num == 0:
                fit_result_dic['f'].append(pixel_dict)
            elif peak_num == 1:
                fit_result_dic['s'].append(pixel_dict)
            else:
                fit_result_dic['m'].append(pixel_dict)

        with open(output_pickle, 'wb') as f:
            pickle.dump(fit_result_dic, f)
        print(f"已儲存 fit 結果分類於 {output_pickle}")
        if n_skipped > 0:
            print(f"已踢除 {n_skipped} 條有分量中心超過 {exclude_range} km/s 的光譜。")
        return fit_result_dic


    def plot_classified_fits(self, fit_result_pickle='fit_result_dic.pickle', category='s', plot_max=30):
        """
        根據分類結果繪圖。category: 's' (單峰), 'm' (多峰), 'f' (失敗)
        """
        import pickle
        import numpy as np
        import matplotlib.pyplot as plt

        # 讀取分類後結果
        with open(fit_result_pickle, 'rb') as f:
            fit_result_dic = pickle.load(f)
        # 讀取原始光譜
        with open(self.input_pickle, 'rb') as f:
            data = pickle.load(f)

        items = fit_result_dic[category]
        n_plotted = 0

        for item in items:
            if plot_max is not None and n_plotted >= plot_max:
                break

            idx = item['index']
            amps = item['amps']
            means = item['means']
            sigmas = item['sigmas']

            if amps is None or means is None or sigmas is None:
                continue
            if len(amps) == 0:
                continue

            idx_peak = np.argmax(amps)
            v_peak = means[idx_peak]

            x = data['x_values'][idx]
            y = data['data_list'][idx]
            
            mask = np.isfinite(x) & np.isfinite(y)

            x = x[mask]
            y = y[mask]

            if len(x) < 3:
                print(f"警告：第 {idx} 條光譜全為 nan/inf 或長度過短，已略過")
                continue


            fit_total = np.zeros_like(x)
            gaussians = []
            for amp, mean, sigma in zip(amps, means, sigmas):
                gauss = amp * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))
                fit_total += gauss
                gaussians.append((gauss, mean, amp))

            plt.figure(figsize=(10, 5), dpi=self.plot_dpi)
            plt.plot(x, y, color='black', lw=1, label='Spectrum')
            plt.plot(x, fit_total, color='red', lw=2, label='Total Fit')

            colors = plt.cm.tab10(np.linspace(0, 1, len(gaussians)))
            for j, (g, mean, amp) in enumerate(gaussians):
                plt.plot(x, g, linestyle='--', color=colors[j % 10], alpha=0.8, label=f'Comp {j+1}')
                plt.axvline(mean, color=colors[j % 10], linestyle='-', linewidth=1)
                plt.text(mean, amp * 1.05, f'v = {mean:.1f} km/s',
                        rotation=90, ha='center', va='bottom', fontsize=8,
                        color=colors[j % 10], backgroundcolor='white')

            plt.title(f'Classified Fit - {category.upper()} (idx={idx})')
            plt.xlabel('Velocity (km/s)')
            plt.ylabel('Intensity')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.legend(fontsize=8, loc='upper right')
            plt.tight_layout()
            plt.show()
            n_plotted += 1

        print(f"Plotted {n_plotted} spectra for category '{category}'.")

    
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


    def stack_restframe(self, plot=True, full_range=True, peak_range=None):
        """
        疊加所有光譜（主峰中心可用 peak_range 篩選），自動產生最大 rest-frame 速度軸。
        peak_range: tuple (vmin, vmax)，僅主峰中心落在此範圍的光譜會被納入
        """
        import pickle
        from scipy.interpolate import interp1d
        import numpy as np
        import matplotlib.pyplot as plt

        with open(self.input_pickle, 'rb') as f:
            data = pickle.load(f)
        with open(self.result_pickle, 'rb') as f:
            result = pickle.load(f)

        x_all = np.array(data['x_values'])
        y_all = np.array(data['data_list'])
        amps_all = np.array(result['amplitudes_fit'], dtype=object)
        means_all = np.array(result['means_fit'], dtype=object)

        # 計算每個主峰位置
        v_peak_arr = np.array([
            means[np.argmax(amps)] if (amps is not None and len(amps) > 0 and means is not None) else np.nan
            for amps, means in zip(amps_all, means_all)
        ])

        # 製作 mask：peak_range 若沒給則全選，有給則依範圍挑選
        if peak_range is not None:
            mask = (~np.isnan(v_peak_arr)) & (v_peak_arr >= peak_range[0]) & (v_peak_arr <= peak_range[1])
        else:
            mask = ~np.isnan(v_peak_arr)

        # 只留下有用的 spectra
        x_all = x_all[mask]
        y_all = y_all[mask]
        v_peak_arr = v_peak_arr[mask]

        # 設定疊加用的 v_grid
        if full_range:
            v_min = np.min([np.nanmin(x) for x in x_all])
            v_max = np.max([np.nanmax(x) for x in x_all])
            v_grid = np.arange(v_min, v_max + self.stack_dv, self.stack_dv)
        else:
            v_grid = np.arange(self.stack_vrange[0], self.stack_vrange[1] + self.stack_dv, self.stack_dv)

        stacked = []

        for x, y, v_peak in zip(x_all, y_all, v_peak_arr):
            x_shifted = x - v_peak
            try:
                f_interp = interp1d(x_shifted, y, kind='linear', bounds_error=False, fill_value=np.nan)
                y_interp = f_interp(v_grid)
                if self.normalize:
                    y_interp = y_interp / np.nanmax(np.abs(y_interp))
                stacked.append(y_interp)
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

        n_total = len(result['amplitudes_fit'])
        n_used = mask.sum()
        print(f'Stacking done: {n_used}/{n_total} spectra used ({100 * n_used / n_total:.1f}%)')
        if peak_range is not None:
            print(f"（僅納入主峰在 {peak_range[0]} ~ {peak_range[1]} km/s 的光譜）")

        return v_grid, mean_spec, std_spec


    def plot_stacked_zoom(
        self,
        v_rest, mean_rest,
        v_raw, mean_raw,
        target_freq,
        freq_raw0,
        vmin=-50, vmax=50,
        c=299792.458
    ):
        """
        Rest-frame 疊加與原始疊加的對比放大顯示，並標註目標頻率對應速度。
        v_rest, mean_rest: rest-frame 疊加結果
        v_raw, mean_raw: 原始疊加結果
        target_freq: 要標註的頻率 (GHz)
        freq_raw0: 疊加後主峰頻率 (GHz)
        vmin, vmax: 欲放大顯示的速度區間
        """
        peak_idx = np.nanargmax(mean_raw)
        v_peak_raw = v_raw[peak_idx]
        v_raw_aligned = v_raw - v_peak_raw

        v_target_raw = c * (1 - target_freq / freq_raw0)
        v_target_raw_aligned = v_target_raw - v_peak_raw

        mask_rest = (v_rest >= vmin) & (v_rest <= vmax)
        mask_raw = (v_raw_aligned >= vmin) & (v_raw_aligned <= vmax)

        plt.figure(figsize=(10, 5))
        plt.plot(v_raw_aligned[mask_raw], mean_raw[mask_raw],
                label='Raw Stacked Spectrum (Aligned)', color='gray', linewidth=1.5)
        plt.plot(v_rest[mask_rest], mean_rest[mask_rest],
                label='Rest-frame Stacked Spectrum', color='black', linewidth=2)
        plt.axvline(0, linestyle='--', color='red', label='Aligned Main Peak (v=0 km/s)')
        plt.axvline(v_target_raw_aligned, linestyle='--', color='orange',
                    label=f'{target_freq:.3f} GHz ({v_target_raw_aligned:.1f} km/s)')
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Stacked Intensity')
        plt.title(f'Zoomed Stacked Spectrum ({vmin} ~ {vmax} km/s)')
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()
        plt.xlim(vmin, vmax)
        plt.show()

    def plot_lines_on_stack(self, v_grid, mean_spec, line_dict, restfreq=None, label_offset=0.02):
        """
        疊加光譜自動標註多個分子線。
        v_grid：疊加後的速度軸
        mean_spec：疊加後的光譜
        line_dict：{'分子名': 頻率GHz, ...}
        restfreq：主峰基準 rest frequency（GHz），預設取 self.restfreq
        """
        c = 299792.458  # km/s

        # 選定 rest frequency（如果沒給就用 self.restfreq）
        if restfreq is None:
            if hasattr(self, "restfreq"):
                restfreq = self.restfreq
            else:
                raise ValueError("請指定 restfreq 或於 class 中設定 self.restfreq。")

        # 主峰對齊（rest-frame）
        peak_idx = np.nanargmax(mean_spec)
        v_peak = v_grid[peak_idx]
        v_aligned = v_grid - v_peak

        plt.figure(figsize=(12,5))
        plt.plot(v_aligned, mean_spec, lw=1.5, label='Stacked Spectrum', color='black')
        plt.axvline(0, ls='--', color='red', label='Main peak (v=0)')

        y_max = np.nanmax(mean_spec)
        for name, freq in line_dict.items():
            v_target = c * (1 - freq / restfreq)
            v_target_aligned = v_target - v_peak
            plt.axvline(v_target_aligned, ls=':', color='blue', alpha=0.7)
            plt.text(v_target_aligned, y_max * (1-label_offset), f"{name}\n{freq:.3f}GHz",
                     rotation=90, va='top', ha='center', fontsize=10, color='blue')
        
        plt.xlabel("Rest-frame Velocity (km/s)")
        plt.ylabel("Stacked Intensity")
        plt.title("Rest-frame Stacked Spectrum with Line Labels")
        
        plt.legend()
        plt.grid(True, ls=':')
        plt.tight_layout()
        plt.show()

        # 回傳每條線的 rest-frame 位置
        return {name: c * (1 - freq / restfreq) - v_peak for name, freq in line_dict.items()}


    def config(self):
        """
        Return current pipeline parameter settings as a dict.
        """
        return self.__dict__

    