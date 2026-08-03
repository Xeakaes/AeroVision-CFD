"""pytest ortak kurulum: birim testler her zaman CPU/NumPy backend'inde calisir.

GPU (cupy/OpenCL) ortamda bile testlerin deterministik numpy davranisi
bekledigi kabul edilir; GPU yollari gpu_selftest.py ile ayri dogrulanir.
"""

import gpu_backend

gpu_backend.detect_gpu("cpu")
