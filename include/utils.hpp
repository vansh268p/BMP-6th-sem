#ifndef UTILS_HPP
#define UTILS_HPP

#include "types.hpp"
#include <sycl/sycl.hpp>

extern double seed;
size_t trim(char *out, size_t len);		///< Function to trim the string trailing spaces
/**
 * @brief Clean up the hash map for the next iteration
 * @param params Grid parameters
 * @param q SYCL queue for device operations
 * @param dmap Device view of hash map to cleanup
 * @param points_device Pointer to points array on the device
 */
void cleanup_map(const GridParams& params,
                 sycl::queue& q,
                 sycl_hashmap::DeviceView& dmap,
                 Point* points_device);

void initUtility();													///< Initializaion of utility functionality
//double ran1();														///< Random number generator 1
double ran2();														///< Random number generator 2
//void valeat(double *vx, double *vy, double *vz, double v);			///< New random velocity generator 1
//void vmaxwn(double *vx, double *vy, double *vz, double v);			///< New random velocity generator 2
void vmaxwn2(double *vx, double *vy, double *vz, double v);			///< New random velocity generator 3


/**
 * @brief Kinetic Energy Calculations
 * @param q SYCL queue for device operations
 * @param Point Pointer to points array on the device
 * @param END_NUM_Points Number of points to consider for energy calculation
 */
double calculate_energy(sycl::queue& q,const Point* point, int END_NUM_Points,const double* CtoM,int ptype);
int calculate_prob_size(sycl::queue &q,sycl_hashmap::DeviceView& dmap,int max_prob_size,int probing_length,const GridParams& params);

//void printEnergy(char *dir, int iteration);

void printTotalEnergy(char *dir, int iteration,double Global_Total_Energy_kin[2],double Global_Total_Energy_pot[2]);

double calc_energy_serial(const Point *points_host,int END_NUM_Points,const double *CtoM,int ptype);


[[sycl::external]] inline int store_deleted_indices(int index, int* empty_space, int* empty_space_count, int n) {
    auto atomic_count = sycl::atomic_ref<int,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>(*empty_space_count);
    int k_idx = atomic_count.fetch_add(1);
    if (k_idx < n) {
        empty_space[k_idx] = index;
        return 1;
    }
    return 0;
}

int compact_particles(Point* points_device,
                       sycl_hashmap::DeviceView& dmap,
                       sycl::queue& q,
                       int* empty_space,
                       int* empty_space_idx,
                       int END_NUM_Points,
                       int* NUM_Points,
                       double& alloc_time,
                       double& scan_time);

int generate_new_electrons(Point* point_device,
                            sycl::queue& q,
                            const GridParams& params,
                            sycl_hashmap::DeviceView& dmap,
                            int END_NUM_Points,
                            int* NUM_Points,
                            int space,
                            double* CtoM,
                            int electronTemp,
                            double& stored,
                            int num_new
                            );

void generate_pairs(Particle& electron, Particle& ion, sycl::queue& q, const GridParams& params);

void printPhi(char *dir,int iteration);
void printElectricField(char *dir,int iteration);
void printEnergy(char *dir, int iteration);
void printNumberDensity(char *dir, int iteration);
void printNumParticles(char *dir, int iteration,int* num_particle,int ptype);
#endif // UTILS_HPP