#include "types.hpp"

// Global variable definitions
int Maxiter = 10;
int num_type = 2;
double tt = 0;
int prob_size = 32;
int working_size = 32;

double *CtoM = nullptr;
double* accessor_s1 = nullptr;
double* accessor_s2 = nullptr;
double* accessor_s3 = nullptr;
double* accessor_s4 = nullptr;
double* corner_mesh = nullptr;
double* accessor_s1_enrgy = nullptr;
double* accessor_s2_enrgy = nullptr;
double* accessor_s3_enrgy = nullptr;
double* accessor_s4_enrgy = nullptr;
double* corner_mesh_enrgy = nullptr;
int* end_idx = nullptr;
int* deleted_mask = nullptr;
int* back_invalid_mask = nullptr;
int* deleted_offset = nullptr;
int* back_offset = nullptr;
// int* empty_space = nullptr;
// int* empty_space_idx = nullptr;

Particle* particles = nullptr;
GridParams params1;

//Unknown Pointers - to me
double *rho = nullptr;					///< Array of charge density on Grid points
double *phi = nullptr;					///< Array of potential on Grid points
double *magneticField = nullptr;		///< Magnetic field on grid points
Field *electricField = nullptr;		///< Electric field on grid points
//double *energy = nullptr;
//double Total_energy_kinetic[2]; 
//double Total_energy_Pot[2]; 
double *rho1 = nullptr;					///< Array of charge density on Grid points
double *phi1 = nullptr;					///< Array of potential on Grid points	
double *energy1 = nullptr;				///< Energy on grid points
Geometry *geometries = nullptr;
double *eledensavg = nullptr; 		///<M---for average electron density in space-time average
double *iondensavg = nullptr;		///<M---for average ion density in space-time average
double *Teavg = nullptr;			///<M---for average ele temp in space-time average
double *Tiavg = nullptr;			///<M---for average ion temp in space-time average	
double *phiavg = nullptr;			///<M---for average potential in space-time average
double* TPENsum = nullptr;             ///<M---for average electron temperature
double* TPENsum1 = nullptr;             ///<M---for average electron temperature
double *weighingFactor = nullptr;			///< weighing factor for each super particle i.e, mass of the super particle(SI unit)
double* particleCharges = nullptr;		///< Array for charges of particle of different type
double* actualParticles = nullptr; 	///< Array for no of Actual Particles of all types
//int* particleStartIndex;		///< Array for start index of particle type
int* simulationParticles = nullptr;		///< Array for no of Simulation Particles of all types
long *initial_simulation_particles = nullptr;


int TIMESTEPS = 10000;			///< No of iterations (simulationTime = TIMESTEPS * dt)
int PARTICLES = 200;			///< No of particles 
//int PARTICLES_BUFFERSIZE = 220; ///< Particle buffer size with auxillary space  
int GRID_X = 101;				///< No of grid points in x direction
int GRID_Y = 101;				///< No of grid points in y direction
//int totalParticleTypes = 2;		///< No of different types of particle
int periodicBoundary = 0;		///< Whether boundary is periodic or not
int isUniformInit = 1;			///< Uniform Initialization or not
int doReinjection = 0;			///< Whether to reinject particle or not(if they go out)
int collisionFlag = 0;
int printInterval = 10;
int STARTprintspaceavg=10;
int ENDprintspaceavg=10;

double GRID_WIDTH = 0.23;			///< Width of whole grid (#Cells in x dimention * width of each cell)	
double GRID_HEIGHT = 0.1;			///< height of whole grid (#Cells in y dimention * height of each cell)

double initFractionLeft = 0.05;		///< If isUniformInit = 0 then left fraction of init box 
double initFractionRight = 0.33;	///< If isUniformInit = 0 then Right fraction of init box 
double initFractionTop = 0.0;		///< If isUniformInit = 0 then Top fraction of init box 
double initFractionBottom = 1.0;	///< If isUniformInit = 0 then Bottom fraction of init box 

double surfArea = 0.24 * 0.1;		///< surface area of GRID

double dt = 0.00000000018; 		///< Time step for simulation

double B0 = 0.003;				///< max magmetic field in Tesla
double sigma = 0.08;			///< width of the initial magnetic field in meter
double mean = 0.7*0.23;			///< A place where max magnetic field occur

double electronTemp = 10;		///< Electron tempetrature in eV
double ionTemp = 0.026;			///< Ion temperature in eV
double gasTemp = 1000;			///< Gas temperature in K

double pressure = 0.690325;		///< Pressure of gas in Pascal

double epsilon = 8.854e-12;					///< epsilon = epsilon_0 * epsilon_r
//double auxiliarySpaceFraction = 1;			///< Extra space for particle addition shared by all species 
double auxiliarySpaceFraction = 40;	
double boltzmannConstant = 1.3806503e-23;	///< Boltzmann constant

//int Iter;                         // Iteration number used for restart mechanism//
double demax = 0;							///< Max electron number density with proper weighing

double x22 = 0.01;
double x11 = 0.0025;
double xm  =  0.00625;
double ionimax;

double ibmax = 150;
double a1 = 0.00779;
double a2 = 0.00918;
double b1 = -0.003489;
double b2 = 0.000798;
double sigma1 =0.00625;

double Jm, Jec1;
int nop;
int Iter = 0;


int geometryCount;
double dx ; 					///< Cell width
double dy ;						///< Cell height


double Total_energy_kinetic[2] = {0.0, 0.0}; 
double Total_energy_Pot[2] = {0.0, 0.0}; 
double Total_energy[2] = {0.0, 0.0};
char gasName[30];

double power;
int powerMode=0;
double heatFrequency;


/* Self-aware sort variables */
double threshFactor=3;		/// Setting the threshold wrt min mover time
double maxIter=-1.0;
double benchTime=10000000.0;
double iTime=0.0;
double threshTime=-1.0;

double enHeat_global=0.0;
double DEPS_global=0.0;
//inline constexpr double p_val = 0.1;

int scaleFactor;
double attachmentFactor = 1;
double stored = 0;
double dng;