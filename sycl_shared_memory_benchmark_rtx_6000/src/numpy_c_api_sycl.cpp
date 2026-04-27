#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "sycl_shared_core.h"

#include <chrono>

static long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double ns_ms(long long ns) {
    return static_cast<double>(ns) / 1.0e6;
}

static PyObject* wrap_ptr(float* ptr, size_t n) {
    npy_intp dims[2] = {static_cast<npy_intp>(n), static_cast<npy_intp>(n)};
    PyObject* arr = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT32, ptr);
    if (arr) {
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(arr), NPY_ARRAY_OWNDATA);
    }
    return arr;
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
    PyObject* a = wrap_ptr(sycl_shared_a(), n);
    PyObject* b = wrap_ptr(sycl_shared_b(), n);
    PyObject* c = wrap_ptr(sycl_shared_c(), n);
    PyObject* out = Py_BuildValue("(NNN)", a, b, c);
    return out;
}

static PyObject* py_finalize(PyObject*, PyObject*) {
    sycl_shared_finalize();
    Py_RETURN_NONE;
}

static PyObject* py_matmul(PyObject*, PyObject* args) {
    PyObject *a_obj, *b_obj, *c_obj;
    long long py_call_ns;
    if (!PyArg_ParseTuple(args, "OOOL", &a_obj, &b_obj, &c_obj, &py_call_ns)) return nullptr;
    long long entry_ns = now_ns();
    long long validate0 = now_ns();
    PyArrayObject* a_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(a_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* b_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(b_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* c_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(c_obj, NPY_FLOAT32, NPY_ARRAY_OUT_ARRAY));
    if (!a_arr || !b_arr || !c_arr) {
        Py_XDECREF(a_arr); Py_XDECREF(b_arr); Py_XDECREF(c_arr);
        return nullptr;
    }
    if (PyArray_NDIM(a_arr) != 2 || PyArray_DIM(a_arr, 0) != PyArray_DIM(a_arr, 1)) {
        Py_DECREF(a_arr); Py_DECREF(b_arr); Py_DECREF(c_arr);
        return PyErr_Format(PyExc_ValueError, "expected square 2D arrays");
    }
    size_t n = static_cast<size_t>(PyArray_DIM(a_arr, 0));
    long long validate1 = now_ns();
    SyclSharedTimings t{};
    int rc = sycl_shared_matmul(static_cast<float*>(PyArray_DATA(a_arr)),
                                static_cast<float*>(PyArray_DATA(b_arr)),
                                static_cast<float*>(PyArray_DATA(c_arr)),
                                n, &t);
    Py_DECREF(a_arr); Py_DECREF(b_arr); Py_DECREF(c_arr);
    if (rc != 0) return PyErr_Format(PyExc_RuntimeError, "sycl_shared_matmul failed: %d", rc);
    PyObject* out = PyDict_New();
    PyDict_SetItemString(out, "marshal_in_ms", PyFloat_FromDouble(ns_ms(entry_ns - py_call_ns)));
    PyDict_SetItemString(out, "numpy_validate_ms", PyFloat_FromDouble(ns_ms(validate1 - validate0)));
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

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT, "numpy_c_api_sycl", nullptr, -1, Methods,
};

PyMODINIT_FUNC PyInit_numpy_c_api_sycl(void) {
    import_array();
    return PyModule_Create(&Module);
}
