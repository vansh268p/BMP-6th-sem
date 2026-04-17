/*
 * sycl_pic_ctypes.cpp — Plain C shared-library wrapper for SYCL-PIC.
 *
 * NO Python headers, NO NumPy — just extern "C" functions that expose
 * the simulation API.  Python loads this .so via ctypes.CDLL and
 * wraps returned raw pointers with numpy.ctypeslib.as_array() for
 * zero-copy interop.
 *
 * Compile:
 *   icx -fsycl -fsycl-targets=nvptx64-nvidia-cuda \
 *       -shared -fPIC -O3 -std=c++17 -DMKL_ILP64 \
 *       -I./include -I<mkl>/include \
 *       -L<mkl>/lib -L<tbb>/lib \
 *       -lmkl_sycl_blas ... -lmkl_core -lsycl -lOpenCL -ltbb ... \
 *       src/sycl_pic_ctypes.cpp src/utils.cpp ... \
 *       -o build/libsycl_pic.so
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
#include <cstdio>

#include "types.hpp"
#include "particles.hpp"
#include "mover.hpp"
#include "interpolation.hpp"
#include "utils.hpp"
#include "sycl_hashmap.hpp"
#include "poissonSolver.hpp"

/* ── internal state (same role as main.cpp locals) ── */
static sycl::queue g_queue;
static size_t      g_wg_size     = 0;
static FILE*       g_f_particle  = nullptr;

/* ══════════════════════════════════════════════════════════════
   All exported functions use C linkage (no name mangling).
   ══════════════════════════════════════════════════════════════ */
extern "C" {

/* ── Initialisation ────────────────────────────────────────── */
int sycl_pic_init(const char* dev_type, int wg, int ps,
                  const char* input1, const char* input2)
{
    std::string dt(dev_type);
    if (dt == "cpu")
        g_queue = sycl::queue(sycl::cpu_selector_v);
    else if (dt == "gpu")
        g_queue = sycl::queue(sycl::gpu_selector_v);
    else {
        std::cerr << "sycl_pic_init: device must be 'cpu' or 'gpu'\n";
        return -1;
    }

    g_wg_size    = static_cast<size_t>(wg);
    prob_size    = ps;
    working_size = static_cast<int>(g_wg_size);

    std::vector<std::string> input_files;
    input_files.push_back(std::string(input1));
    input_files.push_back(std::string(input2));

    auto t0 = std::chrono::high_resolution_clock::now();
    initVariables("M_PICDAT.DAT", g_queue, input_files);
    auto t1 = std::chrono::high_resolution_clock::now();

    double total = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Time Taken for Insertion: " << tt << " s\n";
    std::cout << "Time Taken for Particle Registration and Insertion: "
              << total << " s\n";
    std::cout << "Charge to Mass Ratio : "
              << CtoM[0] << " " << CtoM[1] << std::endl;

    /* Poisson solver init — this calls pardisoinit internally */
    initPoissonSolver(g_queue);

    /* Trajectory file */
    g_f_particle = fopen("particle_trajectory.out", "a");

    return 0;   /* success */
}

/* ── Simple getters ────────────────────────────────────────── */
int sycl_pic_get_grid_size(void) {
    return params1.GRID_X * params1.GRID_Y;
}

int sycl_pic_get_timesteps(void) {
    return TIMESTEPS;
}

int sycl_pic_get_print_interval(void) {
    return printInterval;
}

int sycl_pic_get_num_species(void) {
    return num_type;
}

int sycl_pic_get_particle_count(int species) {
    return particles[species].END_NUM_Points;
}

/* ── Zero-copy pointer access ─────────────────────────────── */
double* sycl_pic_get_rho_ptr(void) {
    return rho;
}

double* sycl_pic_get_phi_ptr(void) {
    return phi;
}

/* ── Zero rho ──────────────────────────────────────────────── */
void sycl_pic_zero_rho(void) {
    memset(rho, 0, params1.GRID_X * params1.GRID_Y * sizeof(double));
    memset(TPENsum, 0, num_type * sizeof(double));
}

/* ── Charge deposition ─────────────────────────────────────── */
double sycl_pic_charge_deposition(int species) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int cur_ps = calculate_prob_size(
        g_queue, particles[species].dmap,
        prob_size, prob_size * working_size, params1);
    interpolate(
        particles[species].final_mesh,
        particles[species].energy_mesh,
        rho,
        particles[species].d_points,
        particles[species].params,
        g_queue,
        corner_mesh, corner_mesh_enrgy,
        g_wg_size,
        particles[species].dmap,
        prob_size * working_size, cur_ps,
        accessor_s1, accessor_s2, accessor_s3, accessor_s4,
        accessor_s1_enrgy, accessor_s2_enrgy,
        accessor_s3_enrgy, accessor_s4_enrgy,
        particles[species].END_NUM_Points, species);
    g_queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

/* ── Poisson solver ────────────────────────────────────────── */
double sycl_pic_poisson_solve(int iteration) {
    auto t0 = std::chrono::high_resolution_clock::now();
    poissonSolver(rho, phi, iteration, g_queue);
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

/* ── Mover ─────────────────────────────────────────────────── */
double sycl_pic_move_particles(int species, int iteration) {
    auto t0 = std::chrono::high_resolution_clock::now();
    mover(particles[species].d_points, params1, g_queue,
          particles[species].dmap,
          prob_size * working_size, iteration,
          end_idx, prob_size,
          particles[species].NUM_Points,
          particles[species].empty_space,
          particles[species].empty_space_idx,
          species,
          particles[species].END_NUM_Points,
          particles[species].new_additions);
    g_queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

/* ── Compact particles ─────────────────────────────────────── */
double sycl_pic_compact_particles(int species,
                                  double* out_alloc_t,
                                  double* out_scan_t)
{
    double alloc_t = 0.0, scan_t = 0.0;
    auto t0 = std::chrono::high_resolution_clock::now();
    particles[species].END_NUM_Points = compact_particles(
        particles[species].d_points, particles[species].dmap, g_queue,
        particles[species].empty_space,
        particles[species].empty_space_idx,
        particles[species].END_NUM_Points,
        particles[species].NUM_Points,
        alloc_t, scan_t);
    g_queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();

    if (out_alloc_t) *out_alloc_t = alloc_t;
    if (out_scan_t)  *out_scan_t  = scan_t;
    return std::chrono::duration<double>(t1 - t0).count();
}

/* ── Map cleanup ───────────────────────────────────────────── */
double sycl_pic_cleanup_map(int species) {
    auto t0 = std::chrono::high_resolution_clock::now();
    cleanup_map(params1, g_queue,
                particles[species].dmap,
                particles[species].d_points);
    g_queue.wait_and_throw();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

/* ── Generate new particles ────────────────────────────────── */
double sycl_pic_generate_new_particles(void) {
    auto t0 = std::chrono::high_resolution_clock::now();

    int num_ele = 0, num_ion = 0;
    g_queue.memcpy(&num_ele, particles[0].new_additions, sizeof(int)).wait();
    g_queue.memcpy(&num_ion, particles[1].new_additions, sizeof(int)).wait();
    int num_new = num_ele - num_ion;
    if (num_new < 0) num_new = 0;
    g_queue.memset(particles[0].new_additions, 0, sizeof(int)).wait();
    g_queue.memset(particles[1].new_additions, 0, sizeof(int)).wait();

    particles[0].END_NUM_Points = generate_new_electrons(
        particles[0].d_points, g_queue, params1,
        particles[0].dmap,
        particles[0].END_NUM_Points,
        particles[0].NUM_Points,
        particles[0].space, CtoM, electronTemp, stored, num_new);
    g_queue.wait_and_throw();

    generate_pairs(particles[0], particles[1], g_queue, params1);
    g_queue.wait_and_throw();

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

/* ── Diagnostics ───────────────────────────────────────────── */
void sycl_pic_print_diagnostics(const char* dir, int iteration) {
    printNumberDensity(const_cast<char*>(dir), iteration);
    printEnergy(const_cast<char*>(dir), iteration);
    printElectricField(const_cast<char*>(dir), iteration);
    printPhi(const_cast<char*>(dir), iteration);
}

/* ── Save meshes ───────────────────────────────────────────── */
void sycl_pic_save_meshes(void) {
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
}

/* ── Finalise ──────────────────────────────────────────────── */
void sycl_pic_finalize(void) {
    if (g_f_particle) { fclose(g_f_particle); g_f_particle = nullptr; }
    freeVariables(g_queue);
}

}  /* extern "C" */
