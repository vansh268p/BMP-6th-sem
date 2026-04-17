/*
 * launcher.cpp — Thin C++ executable that embeds Python and runs main.py.
 *
 * This solves the MKL Pardiso segfault issue: MKL Pardiso crashes when
 * called from a shared library loaded via dlopen (whether from Python's
 * import machinery or ctypes.CDLL).  By compiling sycl_pic_module.cpp
 * directly into this executable and registering it as a built-in Python
 * module, all MKL symbols are resolved at build time — no dlopen, no crash.
 *
 * From the user's perspective, main.py IS the simulation driver.
 * This launcher just:
 *   1. Registers the sycl_pic C extension as a built-in module
 *   2. Initialises the embedded Python interpreter
 *   3. Passes command-line args through to main.py
 *   4. Runs main.py
 *
 * Usage:
 *   ./build/sycl_pic_launcher <input1> <input2> <cpu|gpu> <wg_size> <prob_size>
 */

#include <cstdio>
#include <cstdlib>
#include <Python.h>

/* Forward-declare the module init function from sycl_pic_module.cpp */
extern "C" PyObject* PyInit_sycl_pic(void);

int main(int argc, char** argv)
{
    /* 1) Register sycl_pic as a built-in module BEFORE Py_Initialize.
     *    When main.py does "import sycl_pic", Python calls PyInit_sycl_pic
     *    directly from the executable — no dlopen, no .so loading. */
    if (PyImport_AppendInittab("sycl_pic", PyInit_sycl_pic) == -1) {
        fprintf(stderr, "Error: could not register sycl_pic built-in module\n");
        return 1;
    }

    /* 2) Set up Python home from environment for flexible runtime activation */
    wchar_t* wpyhome = nullptr;
    const char* pyhome_env = std::getenv("PYTHONHOME");
    if (pyhome_env && pyhome_env[0] != '\0') {
        wpyhome = Py_DecodeLocale(pyhome_env, nullptr);
        if (!wpyhome) {
            fprintf(stderr, "Error: could not decode PYTHONHOME\n");
            return 1;
        }
        Py_SetPythonHome(wpyhome);
    }

    /* 3) Convert argv to wide-char for PySys_SetArgvEx */
    wchar_t** wargv = new wchar_t*[argc];
    for (int i = 0; i < argc; i++) {
        wargv[i] = Py_DecodeLocale(argv[i], nullptr);
        if (!wargv[i]) {
            fprintf(stderr, "Error: could not decode argument %d\n", i);
            return 1;
        }
    }

    /* 4) Initialise the Python interpreter */
    Py_Initialize();
    if (!Py_IsInitialized()) {
        fprintf(stderr, "Error: Python interpreter failed to initialise\n");
        return 1;
    }

    /* Pass command-line args to sys.argv (0 = don't update sys.path) */
    PySys_SetArgvEx(argc, wargv, 0);

    /* 5) Run src/main.py */
    FILE* fp = fopen("src/main.py", "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open src/main.py\n");
        Py_Finalize();
        return 1;
    }

    int rc = PyRun_SimpleFile(fp, "src/main.py");
    fclose(fp);

    /* 6) Finalise Python */
    Py_Finalize();

    /* 7) Clean up wide-char args */
    for (int i = 0; i < argc; i++)
        PyMem_RawFree(wargv[i]);
    delete[] wargv;
    if (wpyhome)
        PyMem_RawFree(wpyhome);

    return (rc == 0) ? 0 : 1;
}
