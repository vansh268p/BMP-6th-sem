#ifndef TYPES_HPP
#define TYPES_HPP

#include <sycl/sycl.hpp>
#include "sycl_hashmap.hpp"
#include "species_ds.hpp"
// Global constants
extern int Maxiter;
extern int num_type;
extern double tt;
inline constexpr double p_val = 0.1;
extern int prob_size;
extern int working_size;

extern double *CtoM;
extern double* accessor_s1;
extern double* accessor_s2;
extern double* accessor_s3;
extern double* accessor_s4;
extern double* corner_mesh;
extern double* accessor_s1_enrgy;
extern double* accessor_s2_enrgy;
extern double* accessor_s3_enrgy;
extern double* accessor_s4_enrgy;
extern double* corner_mesh_enrgy;
extern int* deleted_mask;
extern int* back_invalid_mask;
extern int* deleted_offset;
extern int* back_offset;
extern int* end_idx;
// extern int* empty_space;
// extern int* empty_space_idx;

extern Particle* particles;
extern GridParams params1;
//Unknown Pointers - to me
extern double *rho;					///< Array of charge density on Grid points
extern double *phi;					///< Array of potential on Grid points
extern double *magneticField;		///< Magnetic field on grid points
extern Field *electricField;		///< Electric field on grid points
//extern double *energy;
extern double Total_energy_kinetic[2]; 
extern double Total_energy_Pot[2];
extern double Total_energy[2];
extern double *rho1;					///< Array of charge density on Grid points
extern double *phi1;					///< Array of potential on Grid points	
extern double *energy1;				///< Energy on grid points
extern Geometry *geometries;
extern double *eledensavg; 		///<M---for average electron density in space-time average
extern double *iondensavg;		///<M---for average ion density in space-time average
extern double *Teavg;			///<M---for average ele temp in space-time average
extern double *Tiavg;			///<M---for average ion temp in space-time average	
extern double *phiavg;			///<M---for average potential in space-time average
extern double* TPENsum;             ///<M---for average electron temperature
extern double* TPENsum1;          ///<M---for average electron temperature
extern double *weighingFactor;			///< weighing factor for each super particle i.e, mass of the super particle(SI unit)
extern double* particleCharges;		///< Array for charges of particle of different type
extern double* actualParticles; 	///< Array for no of Actual Particles of all types
//int* particleStartIndex;		///< Array for start index of particle type
extern int* simulationParticles;		///< Array for no of Simulation Particles of all types
extern long *initial_simulation_particles;


//Unknown Variables - to me
extern double enHeat_global;
extern double DEPS_global;
extern int PARTICLES;			///< No of particles 
extern int GRID_X;				///< No of grid points in x direction
extern int GRID_Y;				///< No of grid points in y direction
extern int NTHR;
extern double GRID_WIDTH;		///< Width of whole grid (#Cells in x dimention * width of each cell)
extern double GRID_HEIGHT;		///< height of whole grid (#Cells in y dimention * height of each cell)

extern double computationTimeCD;
extern int periodicBoundary;	///< Whether boundary is periodic or not
extern int isUniformInit;		///< Uniform Initialization or not
extern int doReinjection;		///< Whether to reinject particle or not(if they go out)
extern int collisionFlag;
extern int printInterval;

extern double initFractionLeft;		///< If isUniformInit = 0 then left fraction of init box 
extern double initFractionRight;	///< If isUniformInit = 0 then Right fraction of init box 
extern double initFractionTop;		///< If isUniformInit = 0 then Top fraction of init box 
extern double initFractionBottom;	///< If isUniformInit = 0 then Bottom fraction of init box 

//extern int totalParticleTypes;	///< No of different types of particle
extern int geometryCount;		///< No of geometry lines in the file

extern double electronTemp;			///< Electron tempetrature in eV
extern double ionTemp;				///< Ion temperature in eV
extern double gasTemp;				///< Gas temperature in K

extern double pressure;				///< Pressure of gas in Pascal

extern double dx;	 			///< Cell width
extern double dy;				///< Cell height
extern double dt; 				///< Time step for simulation
extern double demax;			///< Max electron number density with proper weighing
extern double epsilon;			///< Permitivity of medium
extern double boltzmannConstant;///< Boltzmann constant

extern double x22;                /// max length of ioni ///
extern double x11;                /// min length of ioni ///
extern double xm;                /// average of x1, x2///
extern double ionimax;           /// max of ionization rate///

extern double ibmax;
extern double a1;
extern double a2;
extern double b1;
extern double b2;
extern double sigma1;

extern double Jm;
extern double Jec1;
extern int nop;
extern int Iter;

//Should be defined Global??
extern double surfArea;
extern int STARTprintspaceavg;
extern int ENDprintspaceavg;
extern double power;
extern int powerMode;
extern double heatFrequency;
extern char gasName[30];
extern int TIMESTEPS;
extern double B0;				///< max magmetic field in Tesla
extern double sigma;			///< width of the initial magnetic field in meter
extern double mean;			///< A place where max magnetic field occur
extern int scaleFactor;
extern double attachmentFactor;
extern double stored;
extern double dng; // Gas Density
// Forward declarations for kernel classes
class PartialMeshInitKernel;
class InterpolationKernel;
class ReducePartialMeshesKernel;
class InterReductionKernel;
class InsertKernel;
class initKeyKernel;
class initKernel;
class MoverKernel;
class CleanUpKernel;
class TempKernel;
class InsertKernel_legacy;
class initKeyKernel_legacy;
class initKernel_legacy;
class MoverKernel_Particle;
class EnergyKernel;
class RhoKernel;
class CompactionKernel;
class MaskKernel;
class ScatterKernel;
class newElectronKernel;
class newPairKernel;
class MoverInsertions;
class FindLengthKernel;
#endif // TYPES_HPP