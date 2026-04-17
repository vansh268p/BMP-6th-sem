#ifndef MOVER_HPP
#define MOVER_HPP

#include "types.hpp"
#include <sycl/sycl.hpp>

/**
 * @brief Compute modulo operation for double values
 * @param x Dividend
 * @param y Divisor
 * @return Result of x mod y
 */
double modulo(double x, double y);
/**
 * @brief Random mover function for individual particles
 * @param p Pointer to particle data
 * @param idx Particle index
 * @param iter Current iteration number
 * @return 0 if particle moved, 1 if particle stayed in place
 */

 /**
 * @brief Deterministic Mover
 * @param p Pointer to a Particle
 * @param ptype Particle type
 */
inline void deterministic_mover(Point& p, int ptype,const double *CtoM, const double dt, const Field *electricField, const double* magneticField,const GridParams& params,const double* phi);


/**
 * @brief Vectorized Deterministic Mover using SYCL vectors
 * @param p Pointer to a Particle
 * @param ptype Particle type
 */
inline void deterministic_mover_vec(
        Point        *p,
        int           ptype,
        const double *CtoM,            // q / m lookup table
        double        dt,
        const Field  *electricField,   // still zeros in your code
        const double *magneticField,   //  "
        const GridParams &params);

inline int random_mover(Point *p, int idx, int iter);

/**
 * @brief Move particles based on random probability
 * @param points_device Device pointer to particle array
 * @param params Grid parameters
 * @param q SYCL queue for device operations
 * @param dmap Device view of hash map
 * @param probing_length Probing length parameter
 * @param iter Current iteration number
 * @param end_idx Array to store end indices
 */
void mover(Point *points_device,
           const GridParams& params,
           sycl::queue& q,
           sycl_hashmap::DeviceView& dmap,
           const int probing_length,
           int iter,
           int* end_idx,
           int prob_size,
           int *NUM_Points,
           int* empty_space,
           int* empty_space_idx,
           int ptype,
           int num_particles,
           int* new_additions);


#endif // MOVER_HPP