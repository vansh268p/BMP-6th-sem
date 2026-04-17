#ifndef PARTICLES_HPP
#define PARTICLES_HPP

#include "types.hpp"
#include <sycl/sycl.hpp>

/**
 * @brief Read header information from binary input file
 * @param p Particle reference to store header data
 * @param q SYCL queue for device operations
 */
void readHeader(Particle& p, sycl::queue& q);

/**
 * @brief Read particle data from binary input file
 * @param p Particle reference to store data
 * @param input_file Path to input file
 * @param params Grid parameters
 * @param q SYCL queue for device operations
 */
void readFile(Particle& p, std::string input_file, GridParams params, sycl::queue& q,int j);

/**
 * @brief Register a particle species with all necessary initialization
 * @param pt Particle reference to register
 * @param q SYCL queue for device operations
 * @param prob_size Probing size parameter
 * @param working_size Working group size
 */
void registerParticle_legacy(Particle &pt, sycl::queue& q, int prob_size, int working_size,int j);

/**
 * @brief Register a particle species with all necessary initialization
 * @param pt Particle reference to register
 * @param q SYCL queue for device operations
 * @param prob_size Probing size parameter
 * @param working_size Working group size
 */
void registerParticle(Particle &pt, sycl::queue& q, int prob_size, int working_size,std::string input_files,int j);

/**
 * @brief Create Particles and its velocity randomly
 * @param host_points Vector to store generated points
 * @param END_NUM_Points Number of points to generate
 * @param j Particle type index
 */
void initParticlesRandomly(std::vector<Point>& host_points,int END_NUM_Points,int j);
/**
 * @brief Initialize Variables for the simulation
 * @param params Grid parameters
 * @param arr Array of Particle species
 * @param q SYCL queue for device operations
 */
void initVariables_legacy(GridParams& params, std::vector<Particle>& arr, sycl::queue& q);


/**
 * @brief Initialize Variables for the simulation
 * @param PIC_File_Name Path to the PICDAT file
 * @param q SYCL queue for device operations
 */
void initVariables(char* PIC_File_Name, sycl::queue& q,std::vector<std::string>& input_files);
/**
 * @brief Free the allocated space for variables
 */
void freeVariables(sycl::queue& q);

/**
 * @brief initialize charge to mass ratio for given index
 * @param index Index of the particle species
 * @param charge Charge of the particle species
 * @param mass Mass of the particle species
 */
void initCtoM(int index, double charge, double mass);

/**
 * @brief initialize weighing factor for given index
 * @param index Index of the particle species
 * @param simulationParticleCount Number of particles in the simulation
 * @param actualParticleCount Actual number of particles represented
 */
void updateWeighingFactor(int index, int simulationParticleCount, double actualParticleCount);


/**
 * @brief Read the PICDAT file to initialize simulation parameters
 * @param picdatfile Path to the PICDAT file
 * @return Status code (0 for success)
 */
int readPICDAT(char* picdatfile, sycl::queue& q);

/**
 * @brief Read geometry configuration from file
 * @param q SYCL queue for device operations
 */
void readGeometry(sycl::queue& q);

/**
 * @brief Initialize magnetic field as Gaussian with 'width', 'centre' and 'max' defined in the PICDAT
 * @param a
 * @param mean
 * @param stddev
 */
void initMagneticField(double a, double mean, double stddev);

#endif // PARTICLES_HPP




