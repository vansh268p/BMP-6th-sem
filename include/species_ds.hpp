#include "sycl_hashmap.hpp"

/**
 * @brief Grid parameters for simulation domain
 */
struct GridParams {
    int NX, NY, GRID_X, GRID_Y; /*END_NUM_Points*/
    double dx, dy;
    double GRID_WIDTH, GRID_HEIGHT;
};
/**
 * @brief Particle species data structure
 */
struct Particle {
    char name[21];
    char input_file[21];     // original binary input file path (points + header)
    GridParams params;
    Point* d_points = nullptr;  // device pointer
    sycl_hashmap::DeviceView dmap; // device view (copy)
    int capacity = 0;
    double charge_sign = +1.0; // +1 or -1 depending on how they deposit
    int prob_size = 24;
    double* final_mesh;
    double* energy_mesh;
    int END_NUM_Points;
    int* NUM_Points;
    int space;
    int* empty_space;
    int* empty_space_idx;
    int* new_additions;
    int* mover_left;
    int* mover_left_idx;
    // probing partition per species (same default as your code)
};

