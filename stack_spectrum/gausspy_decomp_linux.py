from gausspy_pipeline import GausspyPipeline
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pickle
from astropy.io import fits

# 1. Create pipeline and initialize all parameters
pipeline = GausspyPipeline(
    cube_file='../datacubes/spw0.fits',
    v1=10, v2=1500, x1=190, x2=210, y1=220, y2=240,
    alpha1=0.1, alpha2=12.0, snr_thresh=3.0,
    stack_vrange=(-200, 200), stack_dv=0.2
)
# transform the fits cube to gausspy format
pipeline.prepare_input()

# d = pickle.load(open('spectrum_for_gausspy.pickle', 'rb'))
# print(d.keys())
# print(len(d['x_values']), len(d['data_list']), len(d['errors']))
# print(type(d['x_values'][0]), type(d['data_list'][0]), type(d['errors'][0]))
# print(d['x_values'][0].shape, d['data_list'][0].shape, d['errors'][0].shape)

# for i, spec in enumerate(d['data_list']):
#     if np.any(np.isnan(spec)):
#         print(f"Spectrum {i} 有 NaN！")
#     elif np.any(np.isinf(spec)):
#         print(f"Spectrum {i} 有 Inf！")
#     elif np.all(spec == 0):
#         print(f"Spectrum {i} 全部為 0！")
#     elif len(spec) == 0:
#         print(f"Spectrum {i} 長度為 0！")
#     elif not np.issubdtype(spec.dtype, np.floating):
#         print(f"Spectrum {i} 不是 float 類型！")


# 2. Decomposition
pipeline.run_decomposition()

# 3. Classify fit results
fit_result_dic = pipeline.classify_fit_results('fit_result_dic.pickle')

# 4. Show fit results
with open('fit_result_dic.pickle', 'rb') as f:
    result = pickle.load(f)

print(result.keys())  # 應該會看到 dict_keys(['s', 'm', 'f'])

print(f"單峰（s）數量: {len(result['s'])}")
print(f"多峰（m）數量: {len(result['m'])}")
print(f"擬合失敗（f）數量: {len(result['f'])}")
