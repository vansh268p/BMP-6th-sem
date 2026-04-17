#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include "types.hpp"
#include <sycl/sycl.hpp>


int calculate_working_size(int prob_size,int probing);
/**
 * @brief Perform charge deposition interpolation
 * @param final_mesh Output mesh for charge deposition
 * @param energy_mesh Output mesh for energy deposition
 * @param rho Output charge density mesh
 * @param points_device Device pointer to particle array
 * @param params Grid parameters
 * @param q SYCL queue for device operations
 * @param corner_mesh Temporary corner mesh for calculations
 * @param corner_mesh_enrgy Temporary corner energy mesh for calculations
 * @param wg_size Work group size
 * @param dmap Device view of hash map
 * @param probing_length Probing length parameter
 * @param prob_size Probing size parameter
 * @param accessor_s1 Temporary accessor for corner 1
 * @param accessor_s2 Temporary accessor for corner 2
 * @param accessor_s3 Temporary accessor for corner 3
 * @param accessor_s4 Temporary accessor for corner 4
 * @param accessor_s1_enrgy Temporary accessor for corner energy 1
 * @param accessor_s2_enrgy Temporary accessor for corner energy 2
 * @param accessor_s3_enrgy Temporary accessor for corner energy 3
 * @param accessor_s4_enrgy Temporary accessor for corner energy 4
 */
void interpolate(double* final_mesh,
                 double* energy_mesh,
                 double* rho,
                 const Point* points_device,
                 const GridParams& params,
                 sycl::queue& q,
                 double* corner_mesh,
                 double* corner_mesh_enrgy,
                 size_t wg_size,
                 sycl_hashmap::DeviceView& dmap,
                 const int probing_length,
                 const int prob_size,
                 double* accessor_s1,
                 double* accessor_s2,
                 double* accessor_s3,
                 double* accessor_s4,
                 double* accessor_s1_enrgy,
                 double* accessor_s2_enrgy,
                 double* accessor_s3_enrgy,
                 double* accessor_s4_enrgy,
                 int END_NUM_Points,
                 int ptype);

#endif // INTERPOLATION_HPP