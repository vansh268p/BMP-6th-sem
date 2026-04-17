/*
 * sycl_pic_module.cpp — Python C extension module for SYCL-PIC simulation.
 *
 * Exposes all simulation functions to Python. Arrays like rho/phi are
 * returned as NumPy arrays via PyArray_SimpleNewFromData (zero-copy).
 * The SYCL queue and all global state are managed internally.
 */

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "types.hpp"
#include "particles.hpp"
#include "mover.hpp"
#include "interpolation.hpp"
#include "utils.hpp"
#include "sycl_hashmap.hpp"
#include "poissonSolver.hpp"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* ── internal state ── */
static sycl::queue g_queue;
static size_t      g_wg_size  = 0;
static FILE*       g_f_particle = nullptr;

/* ══════════════════════════════════════════════════════════════
   Helper: wrap a raw double* as a 1-D NumPy array (zero-copy)
   ══════════════════════════════════════════════════════════════ */
static PyObject* wrap_double_ptr(double* ptr, npy_intp len) {
    npy_intp dims[1] = { len };
    PyObject* arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, static_cast<void*>(ptr));
    if (arr) {
        /* Python must NOT free this memory — it belongs to SYCL */
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(arr), NPY_ARRAY_OWNDATA);
    }
    return arr;
}

/* ══════════════════════════════════════════════════════════════
   init(device, wg_size, prob_sz, input1, input2)
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_init(PyObject* /*self*/, PyObject* args) {
    const char* dev_type;
    int wg, ps;
    const char* input1;
    const char* input2;

    if (!PyArg_ParseTuple(args, "siiss", &dev_type, &wg, &ps, &input1, &input2))
        return nullptr;

    /* SYCL queue */
    std::string dt(dev_type);
    if (dt == "cpu")
        g_queue = sycl::queue(sycl::cpu_selector_v);
    else if (dt == "gpu")
        g_queue = sycl::queue(sycl::gpu_selector_v);
    else {
        PyErr_SetString(PyExc_ValueError, "device must be 'cpu' or 'gpu'");
        return nullptr;
    }

    g_wg_size   = static_cast<size_t>(wg);
    prob_size    = ps;
    working_size = static_cast<int>(g_wg_size);

    /* Initialise particles, grid, etc. */
    std::vector<std::string> input_files;
    input_files.push_back(std::string(input1));
    input_files.push_back(std::string(input2));

    auto t0 = std::chrono::high_resolution_clock::now();
    initVariables("M_PICDAT.DAT", g_queue, input_files);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Time Taken for Insertion: " << tt << " s\n";
    std::cout << "Time Taken for Particle Registration and Insertion: " << total << " s\n";
    std::cout << "Charge to Mass Ratio : " << CtoM[0] << " " << CtoM[1] << std::endl;

    /* Poisson solver init */
    initPoissonSolver(g_queue);

    /* Trajectory file */
    g_f_particle = fopen("particle_trajectory.out", "a");

    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════
   get_grid_size() → int
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_get_grid_size(PyObject*, PyObject*) {
    return PyLong_FromLong(params1.GRID_X * params1.GRID_Y);
}

/* ══════════════════════════════════════════════════════════════
   get_grid_x() → int
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_get_grid_x(PyObject*, PyObject*) {
    return PyLong_FromLong(params1.GRID_X);
}

/* ══════════════════════════════════════════════════════════════
   get_grid_y() → int
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_get_grid_y(PyObject*, PyObject*) {
    return PyLong_FromLong(params1.GRID_Y);
}

/* ══════════════════════════════════════════════════════════════
   get_timesteps() → int
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_get_timesteps(PyObject*, PyObject*) {
    return PyLong_FromLong(TIMESTEPS);
}

/* ══════════════════════════════════════════════════════════════
   get_print_interval() → int
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_get_print_interval(PyObject*, PyObject*) {
    return PyLong_FromLong(printInterval);
}

/* ══════════════════════════════════════════════════════════════
   get_num_species() → int
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_get_num_species(PyObject*, PyObject*) {
    return PyLong_FromLong(num_type);
}

/* ══════════════════════════════════════════════════════════════
   get_particle_counts() → (end_e, end_i)
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_get_particle_counts(PyObject*, PyObject*) {
    return Py_BuildValue("(ii)",
        particles[0].END_NUM_Points,
        particles[1].END_NUM_Points);
}

/* ══════════════════════════════════════════════════════════════
   get_rho() → numpy array (zero-copy)
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_get_rho(PyObject*, PyObject*) {
    npy_intp sz = params1.GRID_X * params1.GRID_Y;
    return wrap_double_ptr(rho, sz);
}

/* ══════════════════════════════════════════════════════════════
   get_phi() → numpy array (zero-copy)
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_get_phi(PyObject*, PyObject*) {
    npy_intp sz = params1.GRID_X * params1.GRID_Y;
    return wrap_double_ptr(phi, sz);
}

/* ══════════════════════════════════════════════════════════════
   zero_rho()  — memset rho to 0
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_zero_rho(PyObject*, PyObject*) {
    memset(rho, 0, params1.GRID_X * params1.GRID_Y * sizeof(double));
    memset(TPENsum, 0, num_type * sizeof(double));
    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════
   charge_deposition(species_idx) → elapsed seconds
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_charge_deposition(PyObject*, PyObject* args) {
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) return nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();
    int cur_ps = calculate_prob_size(g_queue, particles[i].dmap,
                                     prob_size, prob_size * working_size, params1);
    interpolate(particles[i].final_mesh, particles[i].energy_mesh, rho,
                particles[i].d_points, particles[i].params, g_queue,
                corner_mesh, corner_mesh_enrgy, g_wg_size, particles[i].dmap,
                prob_size * working_size, cur_ps,
                accessor_s1, accessor_s2, accessor_s3, accessor_s4,
                accessor_s1_enrgy, accessor_s2_enrgy, accessor_s3_enrgy, accessor_s4_enrgy,
                particles[i].END_NUM_Points, i);
    g_queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    return PyFloat_FromDouble(std::chrono::duration<double>(t1 - t0).count());
}

/* ══════════════════════════════════════════════════════════════
   poisson_solve(iteration) → elapsed seconds
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_poisson_solve(PyObject*, PyObject* args) {
    int iter;
    if (!PyArg_ParseTuple(args, "i", &iter)) return nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();
    poissonSolver(rho, phi, iter, g_queue);
    auto t1 = std::chrono::high_resolution_clock::now();

    return PyFloat_FromDouble(std::chrono::duration<double>(t1 - t0).count());
}

/* ══════════════════════════════════════════════════════════════
   update_fields_from_phi() — enforce boundary and recompute E field
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_update_fields_from_phi(PyObject*, PyObject*) {
    overwritePhi(geometries, phi);
    if (periodicBoundary)
        calculateElectricFieldPeriodic(phi, electricField);
    else
        calculateElectricField(phi, electricField);
    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════
   move_particles(species_idx, iteration) → elapsed seconds
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_move_particles(PyObject*, PyObject* args) {
    int i, iter;
    if (!PyArg_ParseTuple(args, "ii", &i, &iter)) return nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();
    mover(particles[i].d_points, params1, g_queue, particles[i].dmap,
          prob_size * working_size, iter, end_idx, prob_size,
          particles[i].NUM_Points, particles[i].empty_space,
          particles[i].empty_space_idx, i,
          particles[i].END_NUM_Points, particles[i].new_additions);
    g_queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    return PyFloat_FromDouble(std::chrono::duration<double>(t1 - t0).count());
}

/* ══════════════════════════════════════════════════════════════
   compact_particles_py(species_idx) → (elapsed, alloc_t, scan_t)
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_compact_particles(PyObject*, PyObject* args) {
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) return nullptr;

    double alloc_t = 0, scan_t = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    particles[i].END_NUM_Points = compact_particles(
        particles[i].d_points, particles[i].dmap, g_queue,
        particles[i].empty_space, particles[i].empty_space_idx,
        particles[i].END_NUM_Points, particles[i].NUM_Points,
        alloc_t, scan_t);
    g_queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    return Py_BuildValue("(ddd)",
        std::chrono::duration<double>(t1 - t0).count(), alloc_t, scan_t);
}

/* ══════════════════════════════════════════════════════════════
   cleanup_map_py(species_idx) → elapsed seconds
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_cleanup_map(PyObject*, PyObject* args) {
    int i;
    if (!PyArg_ParseTuple(args, "i", &i)) return nullptr;

    auto t0 = std::chrono::high_resolution_clock::now();
    cleanup_map(params1, g_queue, particles[i].dmap, particles[i].d_points);
    g_queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    return PyFloat_FromDouble(std::chrono::duration<double>(t1 - t0).count());
}

/* ══════════════════════════════════════════════════════════════
   generate_new_particles() → elapsed seconds
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_generate_new_particles(PyObject*, PyObject*) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int num_ele = 0, num_ion = 0;
    g_queue.memcpy(&num_ele, particles[0].new_additions, sizeof(int)).wait();
    g_queue.memcpy(&num_ion, particles[1].new_additions, sizeof(int)).wait();
    int num_new = num_ele - num_ion;
    if (num_new < 0) num_new = 0;
    g_queue.memset(particles[0].new_additions, 0, sizeof(int)).wait();
    g_queue.memset(particles[1].new_additions, 0, sizeof(int)).wait();

    particles[0].END_NUM_Points = generate_new_electrons(
        particles[0].d_points, g_queue, params1, particles[0].dmap,
        particles[0].END_NUM_Points, particles[0].NUM_Points,
        particles[0].space, CtoM, electronTemp, stored, num_new);
    g_queue.wait_and_throw();

    generate_pairs(particles[0], particles[1], g_queue, params1);
    g_queue.wait_and_throw();

    auto t1 = std::chrono::high_resolution_clock::now();
    return PyFloat_FromDouble(std::chrono::duration<double>(t1 - t0).count());
}

/* ══════════════════════════════════════════════════════════════
   print_diagnostics(directory, iteration)
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_print_diagnostics(PyObject*, PyObject* args) {
    const char* dir;
    int iter;
    if (!PyArg_ParseTuple(args, "si", &dir, &iter)) return nullptr;

    printNumberDensity(const_cast<char*>(dir), iter);
    printEnergy(const_cast<char*>(dir), iter);
    printElectricField(const_cast<char*>(dir), iter);
    printPhi(const_cast<char*>(dir), iter);

    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════
   save_meshes() — write Mesh_0.out, Mesh_1.out
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_save_meshes(PyObject*, PyObject*) {
    for (int i = 0; i < num_type; i++) {
        std::string file = "Mesh_" + std::to_string(i) + ".out";
        std::ofstream outfile(file);
        if (!outfile.is_open()) {
            std::cerr << "Error opening output file: " << file << std::endl;
            continue;
        }
        for (int y = 0; y < particles[i].params.GRID_Y; ++y) {
            for (int x = 0; x < particles[i].params.GRID_X; ++x) {
                outfile << particles[i].final_mesh[
                    static_cast<size_t>(y) * params1.GRID_X + x] << " ";
            }
            outfile << "\n";
        }
        outfile.close();
    }
    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════
   dump_rho_phi_cpp(prefix) — dump rho/phi from C++ side to binary files
   Used to verify zero-copy: compare these with Python-side dumps.
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_dump_rho_phi_cpp(PyObject*, PyObject* args) {
    const char* prefix;
    if (!PyArg_ParseTuple(args, "s", &prefix)) return nullptr;

    npy_intp sz = params1.GRID_X * params1.GRID_Y;

    std::string rho_file = std::string(prefix) + "_rho_cpp.bin";
    std::string phi_file = std::string(prefix) + "_phi_cpp.bin";

    FILE* f;
    f = fopen(rho_file.c_str(), "wb");
    if (f) { fwrite(rho, sizeof(double), sz, f); fclose(f); }
    else { std::cerr << "Cannot write " << rho_file << std::endl; }

    f = fopen(phi_file.c_str(), "wb");
    if (f) { fwrite(phi, sizeof(double), sz, f); fclose(f); }
    else { std::cerr << "Cannot write " << phi_file << std::endl; }

    std::cout << "[CPP-DUMP] Wrote rho (" << sz << " doubles) to " << rho_file << std::endl;
    std::cout << "[CPP-DUMP] Wrote phi (" << sz << " doubles) to " << phi_file << std::endl;

    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════
   finalize() — free SYCL resources
   ══════════════════════════════════════════════════════════════ */
static PyObject* py_finalize(PyObject*, PyObject*) {
    if (g_f_particle) { fclose(g_f_particle); g_f_particle = nullptr; }
    freeVariables(g_queue);
    Py_RETURN_NONE;
}

/* ══════════════════════════════════════════════════════════════
   Method table
   ══════════════════════════════════════════════════════════════ */
static PyMethodDef SyclPicMethods[] = {
    {"init",                   py_init,                   METH_VARARGS, "Initialise simulation"},
    {"get_grid_size",          py_get_grid_size,          METH_NOARGS,  "Return GRID_X * GRID_Y"},
    {"get_grid_x",             py_get_grid_x,             METH_NOARGS,  "Return GRID_X"},
    {"get_grid_y",             py_get_grid_y,             METH_NOARGS,  "Return GRID_Y"},
    {"get_timesteps",          py_get_timesteps,          METH_NOARGS,  "Return TIMESTEPS"},
    {"get_print_interval",     py_get_print_interval,     METH_NOARGS,  "Return printInterval"},
    {"get_num_species",        py_get_num_species,        METH_NOARGS,  "Return num_type"},
    {"get_particle_counts",    py_get_particle_counts,    METH_NOARGS,  "Return (electrons, ions) counts"},
    {"get_rho",                py_get_rho,                METH_NOARGS,  "Return rho as NumPy array (zero-copy)"},
    {"get_phi",                py_get_phi,                METH_NOARGS,  "Return phi as NumPy array (zero-copy)"},
    {"zero_rho",               py_zero_rho,               METH_NOARGS,  "Zero out rho and TPENsum"},
    {"charge_deposition",      py_charge_deposition,      METH_VARARGS, "Run charge deposition for species i"},
    {"poisson_solve",          py_poisson_solve,          METH_VARARGS, "Run Poisson solver for iteration"},
    {"update_fields_from_phi", py_update_fields_from_phi, METH_NOARGS,  "Recompute electric field from current phi"},
    {"move_particles",         py_move_particles,         METH_VARARGS, "Run mover for species i at iteration"},
    {"compact_particles",      py_compact_particles,      METH_VARARGS, "Compact particles for species i"},
    {"cleanup_map",            py_cleanup_map,            METH_VARARGS, "Clean up hashmap for species i"},
    {"generate_new_particles", py_generate_new_particles, METH_NOARGS,  "Generate new electrons and pairs"},
    {"print_diagnostics",      py_print_diagnostics,      METH_VARARGS, "Print diagnostics to directory"},
    {"save_meshes",            py_save_meshes,            METH_NOARGS,  "Save final mesh output files"},
    {"dump_rho_phi_cpp",       py_dump_rho_phi_cpp,       METH_VARARGS, "Dump rho/phi from C++ side to binary files"},
    {"finalize",               py_finalize,               METH_NOARGS,  "Free all SYCL resources"},
    {nullptr, nullptr, 0, nullptr}
};

/* ══════════════════════════════════════════════════════════════
   Module definition
   ══════════════════════════════════════════════════════════════ */
static struct PyModuleDef sycl_pic_module = {
    PyModuleDef_HEAD_INIT,
    "sycl_pic",                         /* module name */
    "SYCL-PIC particle simulation C extension with zero-copy NumPy interop",
    -1,
    SyclPicMethods
};

PyMODINIT_FUNC PyInit_sycl_pic(void) {
    import_array();   /* NumPy C API init */
    return PyModule_Create(&sycl_pic_module);
}
