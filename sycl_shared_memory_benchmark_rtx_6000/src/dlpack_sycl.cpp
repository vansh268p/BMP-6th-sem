#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <chrono>
#include <cstdint>
#include "sycl_shared_core.h"

typedef enum { kDLCPU = 1 } DLDeviceType;
typedef struct { int code; int bits; int lanes; } DLDataType;
typedef struct { DLDeviceType device_type; int device_id; } DLDevice;
typedef struct {
    void* data;
    DLDevice device;
    int ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
} DLTensor;
struct DLManagedTensor;
typedef void (*DLDeleter)(struct DLManagedTensor*);
typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    DLDeleter deleter;
} DLManagedTensor;
typedef struct {
    DLManagedTensor managed;
    int64_t shape[2];
    int64_t strides[2];
} CapsuleOwner;

static long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double ns_ms(long long ns) { return static_cast<double>(ns) / 1.0e6; }

static void no_delete(DLManagedTensor*) {}

static PyObject* make_capsule(float* ptr, size_t n) {
    CapsuleOwner* owner = new CapsuleOwner{};
    owner->shape[0] = static_cast<int64_t>(n);
    owner->shape[1] = static_cast<int64_t>(n);
    owner->strides[0] = static_cast<int64_t>(n);
    owner->strides[1] = 1;
    owner->managed.dl_tensor.data = ptr;
    owner->managed.dl_tensor.device = {kDLCPU, 0};
    owner->managed.dl_tensor.ndim = 2;
    owner->managed.dl_tensor.dtype = {2, 32, 1};
    owner->managed.dl_tensor.shape = owner->shape;
    owner->managed.dl_tensor.strides = owner->strides;
    owner->managed.dl_tensor.byte_offset = 0;
    owner->managed.manager_ctx = owner;
    owner->managed.deleter = no_delete;
    return PyCapsule_New(&owner->managed, "dltensor", [](PyObject* cap) {
        DLManagedTensor* mt = reinterpret_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
        if (!mt) return;
        delete reinterpret_cast<CapsuleOwner*>(mt->manager_ctx);
    });
}

static int parse(PyObject* cap, float** ptr, size_t* n) {
    DLManagedTensor* mt = reinterpret_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
    if (!mt) return 0;
    DLTensor* t = &mt->dl_tensor;
    if (t->ndim != 2 || t->dtype.code != 2 || t->dtype.bits != 32) return 0;
    *ptr = static_cast<float*>(t->data);
    *n = static_cast<size_t>(t->shape[0]);
    return 1;
}

static PyObject* py_has_gpu(PyObject*, PyObject*) {
    if (sycl_shared_has_gpu()) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
}

static PyObject* py_init(PyObject*, PyObject* args) {
    unsigned long long n;
    if (!PyArg_ParseTuple(args, "K", &n)) return nullptr;
    int rc = sycl_shared_init(static_cast<size_t>(n));
    if (rc != 0) return PyErr_Format(PyExc_RuntimeError, "sycl_shared_init failed: %d", rc);
    return Py_BuildValue("(NNN)",
                         make_capsule(sycl_shared_a(), n),
                         make_capsule(sycl_shared_b(), n),
                         make_capsule(sycl_shared_c(), n));
}

static PyObject* py_finalize(PyObject*, PyObject*) {
    sycl_shared_finalize();
    Py_RETURN_NONE;
}

static PyObject* py_matmul(PyObject*, PyObject* args) {
    PyObject *a_cap, *b_cap, *c_cap;
    long long py_call_ns;
    if (!PyArg_ParseTuple(args, "OOOL", &a_cap, &b_cap, &c_cap, &py_call_ns)) return nullptr;
    long long entry_ns = now_ns();
    long long parse0 = now_ns();
    float *a = nullptr, *b = nullptr, *c = nullptr;
    size_t n = 0, nb = 0, nc = 0;
    if (!parse(a_cap, &a, &n) || !parse(b_cap, &b, &nb) || !parse(c_cap, &c, &nc)) {
        return PyErr_Format(PyExc_ValueError, "invalid DLPack capsule");
    }
    long long parse1 = now_ns();
    SyclSharedTimings t{};
    int rc = sycl_shared_matmul(a, b, c, n, &t);
    if (rc != 0) return PyErr_Format(PyExc_RuntimeError, "sycl_shared_matmul failed: %d", rc);
    PyObject* out = PyDict_New();
    PyDict_SetItemString(out, "marshal_in_ms", PyFloat_FromDouble(ns_ms(entry_ns - py_call_ns)));
    PyDict_SetItemString(out, "dlpack_parse_ms", PyFloat_FromDouble(ns_ms(parse1 - parse0)));
    PyDict_SetItemString(out, "kernel_ms", PyFloat_FromDouble(t.kernel_ms));
    PyDict_SetItemString(out, "sycl_total_ms", PyFloat_FromDouble(t.total_ms));
    return out;
}

static PyMethodDef Methods[] = {
    {"has_gpu", py_has_gpu, METH_NOARGS, nullptr},
    {"init", py_init, METH_VARARGS, nullptr},
    {"finalize", py_finalize, METH_NOARGS, nullptr},
    {"matmul", py_matmul, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef Module = {PyModuleDef_HEAD_INIT, "dlpack_sycl", nullptr, -1, Methods};
PyMODINIT_FUNC PyInit_dlpack_sycl(void) { return PyModule_Create(&Module); }
