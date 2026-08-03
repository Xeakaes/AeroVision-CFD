"""
AeroVision CFD - PyOpenCL GPU Backend (fp32/fp64 otomatik)
Intel/AMD GPU'larda array islemleri icin numpy-uyumlu arayuz.
GPU fp64 desteklemiyorsa tum skalerler otomatik float32'ye cevrilir.
"""

import os
import builtins
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# ── Global state ─────────────────────────────────────────────────────────────
_ctx = None
_queue = None
_device = None
_device_name = None
_has_fp64 = False

float32 = np.float32
float64 = np.float32  # init()'te dogru degeri alir
int32 = np.int32
int64 = np.int32


def init(device_preference="auto"):
    global _ctx, _queue, _device, _device_name, _has_fp64
    global float64, int64

    # AV_OPENCL_ALLOW_CPU=1: GPU'suz ortamda dogrulama icin CPU OpenCL (pocl)
    # cihazlarini da dikkate alir (uretım davranisi degismez).
    allow_cpu = os.environ.get("AV_OPENCL_ALLOW_CPU") == "1"

    all_devices = []
    device_types = (cl.device_type.GPU, cl.device_type.CPU) if allow_cpu else (cl.device_type.GPU,)
    for platform in cl.get_platforms():
        for dt in device_types:
            try:
                devices = platform.get_devices(device_type=dt)
            except cl.RuntimeError:
                continue
            for dev in devices:
                all_devices.append(dev)

    if not all_devices:
        raise RuntimeError("Uygun OpenCL GPU cihazi bulunamadi.")

    def _pick_device(devices, preference):
        if preference == "intel":
            d = next((d for d in devices if "Intel" in d.vendor), None)
        elif preference == "amd":
            # Vendor "Advanced Micro Devices, Inc." -> "AMD" ardışık geçmez;
            # cihaz adı kod adi olabilir (Ellesmere) -> "Radeon" da yoktur.
            d = next(
                (
                    d
                    for d in devices
                    if "AMD" in d.vendor.upper()
                    or "ADVANCED MICRO" in d.vendor.upper()
                    or "RADEON" in d.name.upper()
                ),
                None,
            )
        elif preference == "nvidia":
            d = next((d for d in devices if "NVIDIA" in d.vendor), None)
        elif preference == "auto":
            d = devices[0] if devices else None
        else:
            d = None
        if d is None and allow_cpu:
            # Test modu: vendor eslesmesi yerine en buyuk cihazi kullan
            d = builtins.max(devices, key=lambda x: x.global_mem_size)
        return d

    _device = _pick_device(all_devices, device_preference)
    if _device is None:
        raise RuntimeError(f"OpenCL GPU cihazi bulunamadi (tercih: {device_preference}).")

    _ctx = cl.Context([_device])
    _queue = cl.CommandQueue(_ctx)
    _device_name = _device.name
    _has_fp64 = "cl_khr_fp64" in _device.extensions

    if _has_fp64:
        float64 = np.float64
        int64 = np.int64
    else:
        float64 = np.float32
        int64 = np.int32

    print(f"[OpenCL] Cihaz: {_device_name} ({_device.vendor})")
    print(
        f"[OpenCL] fp64: {_has_fp64}, CU: {_device.max_compute_units}, RAM: {_device.global_mem_size / 1024**3:.1f} GB"
    )


def is_initialized():
    return _ctx is not None


def get_device_name():
    return _device_name or "Yok"


def get_has_fp64():
    return _has_fp64


def _resolve_dtype(dtype):
    if dtype is None:
        return np.float32
    dt = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
    if not _has_fp64 and dt == np.float64:
        return np.float32
    return dt


# ── CLArray wrapper ──────────────────────────────────────────────────────────
# PyOpenCL array'i sarar, tum skalerleri fp64 destek yoksa float32'ye cevirir.


class CLArray:
    __slots__ = ("_arr",)

    def __init__(self, arr):
        self._arr = arr

    def __getattr__(self, name):
        return getattr(self._arr, name)

    @property
    def data(self):
        return self._arr.data

    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def size(self):
        return self._arr.size

    def _w(self, r):
        return CLArray(r) if isinstance(r, cl_array.Array) else r

    def _c(self, o):
        """Skaler float64 ise float32'ye cevir."""
        if not _has_fp64 and isinstance(o, float):
            return np.float32(o)
        if isinstance(o, CLArray):
            return o._arr
        return o

    def __add__(s, o):
        return s._w(s._arr + s._c(o))

    def __radd__(s, o):
        return s._w(s._c(o) + s._arr)

    def __sub__(s, o):
        return s._w(s._arr - s._c(o))

    def __rsub__(s, o):
        return s._w(s._c(o) - s._arr)

    def __mul__(s, o):
        return s._w(s._arr * s._c(o))

    def __rmul__(s, o):
        return s._w(s._c(o) * s._arr)

    def __truediv__(s, o):
        return s._w(s._arr / s._c(o))

    def __rtruediv__(s, o):
        return s._w(s._c(o) / s._arr)

    def __neg__(s):
        return s._w(-s._arr)

    def __abs__(s):
        return s._w(abs(s._arr))

    def __gt__(s, o):
        return s._w(s._arr > s._c(o))

    def __ge__(s, o):
        return s._w(s._arr >= s._c(o))

    def __lt__(s, o):
        return s._w(s._arr < s._c(o))

    def __le__(s, o):
        return s._w(s._arr <= s._c(o))

    def __eq__(s, o):
        return s._w(s._arr == s._c(o))

    def __ne__(s, o):
        return s._w(s._arr != s._c(o))

    def __getitem__(s, i):
        return s._w(s._arr[i])

    def __setitem__(s, i, v):
        if isinstance(v, CLArray):
            v = v._arr
        elif not _has_fp64 and isinstance(v, float):
            v = np.float32(v)
        s._arr[i] = v

    def __len__(s):
        return len(s._arr)

    def __float__(s):
        return float(s._arr.get())

    def __repr__(s):
        return f"CLArray({s._arr})"

    def copy(s):
        return s._w(s._arr.copy())

    def astype(s, dt):
        return s._w(s._arr.astype(dt))

    def ravel(s):
        return s._w(s._arr.ravel())

    def reshape(s, sh):
        return s._w(s._arr.reshape(sh))

    def get(s):
        return s._arr.get()


# ── Array olusturma ──────────────────────────────────────────────────────────


def zeros(shape, dtype=None):
    return CLArray(cl_array.zeros(_queue, shape, dtype=_resolve_dtype(dtype)))


def ones(shape, dtype=None):
    return CLArray(cl_array.to_device(_queue, np.ones(shape, dtype=_resolve_dtype(dtype))))


def full(shape, value, dtype=None):
    dt = _resolve_dtype(dtype)
    return CLArray(cl_array.to_device(_queue, np.full(shape, value, dtype=dt)))


def empty(shape, dtype=None):
    return CLArray(cl_array.empty(_queue, shape, dtype=_resolve_dtype(dtype)))


def array(obj, dtype=None):
    if isinstance(obj, CLArray):
        return CLArray(obj._arr.astype(_resolve_dtype(dtype))) if dtype else obj
    if isinstance(obj, cl_array.Array):
        return CLArray(obj.astype(_resolve_dtype(dtype))) if dtype else CLArray(obj)
    arr = np.asarray(obj)
    if not _has_fp64 and arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    elif arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32)
    if dtype:
        arr = arr.astype(_resolve_dtype(dtype))
    return CLArray(cl_array.to_device(_queue, arr))


def asarray(obj, dtype=None):
    if isinstance(obj, CLArray):
        return CLArray(obj._arr.astype(_resolve_dtype(dtype))) if dtype else obj
    return array(obj, dtype=dtype)


def to_device(arr):
    if isinstance(arr, (CLArray, cl_array.Array)):
        return arr
    return array(arr)


def to_numpy(a):
    if isinstance(a, CLArray):
        return np.asarray(a._arr.get())
    if isinstance(a, cl_array.Array):
        return np.asarray(a.get())
    return np.asarray(a)


def _to_numpy(a):
    return to_numpy(a)


def copy(a):
    return a.copy() if isinstance(a, CLArray) else np.copy(a)


def astype(a, dt):
    if isinstance(a, CLArray):
        return a.astype(dt)
    return np.asarray(a, dtype=_resolve_dtype(dt))


# ── Matematiksel islemler ────────────────────────────────────────────────────

# (expr, ctype) -> derlenmis Program — her cagrida derlemeyi onler
_ELWISE_CACHE: dict[tuple[str, str], cl.Program] = {}


def _elwise_unary(arr, expr):
    a = arr._arr if isinstance(arr, CLArray) else arr
    result = cl_array.empty(_queue, a.shape, dtype=a.dtype)
    ctype = "double" if a.dtype == np.float64 else "float"
    key = (expr, ctype)
    prg = _ELWISE_CACHE.get(key)
    if prg is None:
        prg = cl.Program(
            _ctx,
            f"""
    __kernel void op(__global {ctype} *out, __global {ctype} *in, int n) {{
        int i = get_global_id(0);
        if (i < n) {{ {ctype} x = in[i]; out[i] = {expr}; }}
    }}
    """,
        ).build()
        _ELWISE_CACHE[key] = prg
    prg.op(_queue, (int(np.prod(a.shape)),), None, result.data, a.data, np.int32(a.size))
    return CLArray(result)


def sqrt(a):
    if isinstance(a, CLArray):
        return _elwise_unary(a, "sqrt(x)")
    return np.sqrt(a)


def abs(a):
    if isinstance(a, CLArray):
        return _elwise_unary(a, "fabs(x)")
    return np.abs(a)


def log(a):
    if isinstance(a, CLArray):
        return _elwise_unary(a, "log(x)")
    return np.log(a)


def tanh(a):
    if isinstance(a, CLArray):
        return _elwise_unary(a, "tanh(x)")
    return np.tanh(a)


def sin(a):
    if isinstance(a, CLArray):
        return _elwise_unary(a, "sin(x)")
    return np.sin(a)


def cos(a):
    if isinstance(a, CLArray):
        return _elwise_unary(a, "cos(x)")
    return np.cos(a)


def exp(a):
    if isinstance(a, CLArray):
        return _elwise_unary(a, "exp(x)")
    return np.exp(a)


def clip(a, a_min, a_max):
    if isinstance(a, CLArray):
        ctype = "double" if a._arr.dtype == np.float64 else "float"
        return _elwise_unary(a, f"fmin(fmax(x, ({ctype}){a_min}), ({ctype}){a_max})")
    return np.clip(a, a_min, a_max)


def maximum(a, b):
    a_np = (
        to_numpy(a).astype(np.float32)
        if isinstance(a, (CLArray, cl_array.Array))
        else np.asarray(a, dtype=np.float32)
    )
    b_np = (
        to_numpy(b).astype(np.float32)
        if isinstance(b, (CLArray, cl_array.Array))
        else np.asarray(b, dtype=np.float32)
    )
    return to_device(np.maximum(a_np, b_np))


def minimum(a, b):
    a_np = (
        to_numpy(a).astype(np.float32)
        if isinstance(a, (CLArray, cl_array.Array))
        else np.asarray(a, dtype=np.float32)
    )
    b_np = (
        to_numpy(b).astype(np.float32)
        if isinstance(b, (CLArray, cl_array.Array))
        else np.asarray(b, dtype=np.float32)
    )
    return to_device(np.minimum(a_np, b_np))


def where(cond, x, y):
    if isinstance(cond, (CLArray, cl_array.Array)):
        cond_np = to_numpy(cond).astype(np.float32)
        x_np = (
            to_numpy(x).astype(np.float32)
            if isinstance(x, (CLArray, cl_array.Array))
            else np.asarray(x, dtype=np.float32)
        )
        y_np = (
            to_numpy(y).astype(np.float32)
            if isinstance(y, (CLArray, cl_array.Array))
            else np.asarray(y, dtype=np.float32)
        )
        return to_device(np.where(cond_np > 0.5, x_np, y_np))
    return np.where(cond, x, y)


def logical_or(a, b):
    if isinstance(a, (CLArray, cl_array.Array)) or isinstance(b, (CLArray, cl_array.Array)):
        a_np = (
            to_numpy(a).astype(bool)
            if isinstance(a, (CLArray, cl_array.Array))
            else np.asarray(a, dtype=bool)
        )
        b_np = (
            to_numpy(b).astype(bool)
            if isinstance(b, (CLArray, cl_array.Array))
            else np.asarray(b, dtype=bool)
        )
        return to_device(np.logical_or(a_np, b_np).astype(np.float32))
    return np.logical_or(a, b)


def logical_and(a, b):
    if isinstance(a, (CLArray, cl_array.Array)) or isinstance(b, (CLArray, cl_array.Array)):
        a_np = (
            to_numpy(a).astype(bool)
            if isinstance(a, (CLArray, cl_array.Array))
            else np.asarray(a, dtype=bool)
        )
        b_np = (
            to_numpy(b).astype(bool)
            if isinstance(b, (CLArray, cl_array.Array))
            else np.asarray(b, dtype=bool)
        )
        return to_device(np.logical_and(a_np, b_np).astype(np.float32))
    return np.logical_and(a, b)


# ── Reduction ────────────────────────────────────────────────────────────────


def sum(a):
    if isinstance(a, CLArray):
        return float(np.asarray(cl_array.sum(a._arr).get()))
    if isinstance(a, cl_array.Array):
        return float(np.asarray(cl_array.sum(a).get()))
    return float(np.sum(a))


def max(a):
    if isinstance(a, CLArray):
        return float(np.asarray(cl_array.max(a._arr).get()))
    if isinstance(a, cl_array.Array):
        return float(np.asarray(cl_array.max(a).get()))
    return float(np.max(a))


def min(a):
    if isinstance(a, CLArray):
        return float(np.asarray(cl_array.min(a._arr).get()))
    if isinstance(a, cl_array.Array):
        return float(np.asarray(cl_array.min(a).get()))
    return float(np.min(a))


# ── Aralik / Grid ────────────────────────────────────────────────────────────


def linspace(start, stop, num):
    return np.linspace(float(start), float(stop), int(num)).astype(np.float64)


def arange(start=None, stop=None, step=1.0, dtype=None):
    dt = _resolve_dtype(dtype) if dtype else np.float64
    return np.arange(float(start), float(stop), float(step)).astype(dt)


# ── Sekil islemleri ──────────────────────────────────────────────────────────


def reshape(a, shape):
    if isinstance(a, CLArray):
        return CLArray(a._arr.reshape(shape))
    if isinstance(a, cl_array.Array):
        return CLArray(a.reshape(shape))
    return np.reshape(a, shape)


def ravel(a):
    if isinstance(a, CLArray):
        return CLArray(a._arr.ravel())
    if isinstance(a, cl_array.Array):
        return CLArray(a.ravel())
    return np.ravel(a)


def ascontiguousarray(a):
    return a
