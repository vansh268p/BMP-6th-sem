#include "particles.hpp"
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <sycl/sycl.hpp>
#include "sycl_hashmap.hpp"

void readHeader(Particle& p, sycl::queue& q)
{
    std::ifstream infile(p.input_file, std::ios::binary);
    if (!infile.is_open()) {
        std::cerr << "Error opening input file: " << p.input_file << std::endl;
        return;
    }
    infile.read(reinterpret_cast<char*>(&p.params.NX), sizeof(int));
    infile.read(reinterpret_cast<char*>(&p.params.NY), sizeof(int));
    infile.read(reinterpret_cast<char*>(&p.END_NUM_Points), sizeof(int));
    infile.read(reinterpret_cast<char*>(&Maxiter), sizeof(int));
    infile.close();

    if (p.params.NX <= 0 || p.params.NY <= 0 || p.END_NUM_Points < 0 || Maxiter <= 0) {
        std::cerr << "Invalid parameters from input file." << std::endl;
        return;
    }
    p.NUM_Points = sycl::malloc_shared<int>(1, q);
    *(p.NUM_Points) = 0;
    p.params.GRID_X = p.params.NX + 1;
    p.params.GRID_Y = p.params.NY + 1;
    p.params.dx = 1.0 / static_cast<double>(p.params.NX);
    p.params.dy = 1.0 / static_cast<double>(p.params.NY);
}

void readFile(Particle& p, std::string input_file, GridParams params, sycl::queue& q,int j)
{
    std::vector<PointOld> host_points_old;
    std::vector<Point> host_points;
    host_points_old.resize(p.END_NUM_Points);

    // Re-open to read points after params are set
    std::ifstream infile_points(input_file, std::ios::binary);
    infile_points.seekg(3 * sizeof(int) + sizeof(int)); // Seek past NX, NY, END_NUM_Points, Maxiter
    infile_points.read(reinterpret_cast<char*>(host_points_old.data()),
                      p.END_NUM_Points * sizeof(PointOld));
    infile_points.close();

    for(int i = 0; i < p.END_NUM_Points; i++)
    {
        PointOld pold = host_points_old[i];
        Point pnew;
        pnew.x = pold.x;
        pnew.y = pold.y;
        pnew.pointer = -1;
        pnew.z = 0;
		double v = 0, vx = 0, vy = 0, vz = 0;
		if(j == 0) {
			v = sqrt(2 * fabs(CtoM[j]) * electronTemp);
		}
		if(j == 1) {
			v = sqrt(2 * fabs(CtoM[j]) * ionTemp);	
		}

		vmaxwn2(&vx, &vy, &vz, v);
		pnew.vx = vx;
		pnew.vy = vy;
		pnew.vz = vz;

        host_points.push_back(pnew);
    }

    Point* d_points = sycl::malloc_device<Point>(p.END_NUM_Points, q);
    q.memcpy(d_points, host_points.data(), p.END_NUM_Points * sizeof(Point)).wait();
    p.d_points = d_points;
}

void registerParticle_legacy(Particle &pt, sycl::queue& q, int prob_size, int working_size,int j)
{
    readHeader(pt, q);
    readFile(pt, pt.input_file, pt.params, q,j);

    pt.capacity = pt.params.GRID_X * pt.params.GRID_Y * 1024 * 1.2;
    pt.prob_size = prob_size;

    sycl_hashmap::SyclHashMap* hashmap = new sycl_hashmap::SyclHashMap(pt.capacity, q, prob_size*working_size, pt.params.GRID_X*pt.params.GRID_Y);
    pt.dmap = hashmap->device_view();

    q.parallel_for<initKernel_legacy>(
        sycl::range<1>(pt.capacity),
        [=](sycl::id<1> item) {
            const size_t i = item[0];
            pt.dmap.keys[i] = -1;
            pt.dmap.values[i] = -1;
        }
    ).wait();

    q.parallel_for<initKeyKernel_legacy>(
        sycl::range<1>(pt.params.GRID_X*pt.params.GRID_Y),
        [=](sycl::id<1> item) {
            const size_t i = item[0];
            pt.dmap.key_idx[i] = 0;
            pt.dmap.delete_idx[i] = -1;
        }
    ).wait();

    auto start = std::chrono::high_resolution_clock::now();
    q.submit([&](sycl::handler &h) {
        h.parallel_for<InsertKernel_legacy>(sycl::range<1>(pt.END_NUM_Points), [=](sycl::id<1> idx) {
            Point p = pt.d_points[idx];
            const double inv_dx = 1.0 / pt.params.dx;
            const double inv_dy = 1.0 / pt.params.dy;
            const int gx = static_cast<int>(p.x * inv_dx);
            const int gy = static_cast<int>(p.y * inv_dy);
            int grid_key = static_cast<size_t>(gy) * pt.params.GRID_X + gx;
            pt.d_points[idx].pointer = pt.dmap.insert(grid_key, static_cast<int>(idx[0]),pt.NUM_Points);
        });
    }).wait();

    pt.final_mesh = sycl::malloc_shared<double>(static_cast<size_t>(pt.params.GRID_X) * pt.params.GRID_Y, q);

    auto end = std::chrono::high_resolution_clock::now();
    tt += std::chrono::duration<double>(end - start).count();
}
void initParticlesRandomly(std::vector<Point>& host_points,int END_NUM_Points,int j){

	int i;
	//FILE* fe = fopen("./Particle_Energy.out","a");
	for(i=0;i < END_NUM_Points;i++){

		host_points[i].x = (initFractionLeft * GRID_WIDTH) + ((initFractionRight - initFractionLeft) * GRID_WIDTH) * (((double)std::rand()) / ((double)RAND_MAX + 1.0));//(initFractionLeft * params1.GRID_WIDTH) + ((initFractionRight - initFractionLeft) * params1.GRID_WIDTH) * ran2();
		host_points[i].y = (initFractionTop * GRID_HEIGHT) + ((initFractionBottom - initFractionTop) * GRID_HEIGHT) * (((double)std::rand()) / ((double)RAND_MAX + 1.0)); //(initFractionTop * params1.GRID_HEIGHT) + ((initFractionBottom - initFractionTop) * params1.GRID_HEIGHT) * ran2();
		host_points[i].z = 0;

		if(isUniformInit){
			host_points[i].x = GRID_WIDTH * ((double)std::rand()) / ((double)RAND_MAX + 1.0);
			host_points[i].y = GRID_HEIGHT * ((double)std::rand()) / ((double)RAND_MAX + 1.0);
		}

		double v = 0, vx = 0, vy = 0, vz = 0;
		if(j == 0) {
			v = sqrt(2 * fabs(CtoM[j]) * electronTemp);
		}
		if(j == 1) {
			v = sqrt(2 * fabs(CtoM[j]) * ionTemp);	
		}

		//vmaxwn2(&vx, &vy, &vz, v);
		double z1 = sqrt(-log(ran2()));
		double z2 = 6.283185307 * ran2();
		vx = v * z1 * cos(z2);
		vy = v * z1 * sin(z2);
		if(stored == 0){
			z1 = sqrt(-log(ran2()));
			z2 = 6.283185307 * ran2();
			vz = v * z1 * cos(z2);
			stored = z1 * sin(z2);
		}
		else{
			vz = v * stored;
			stored = 0;
		}
		host_points[i].vx = vx;
		host_points[i].vy = vy;
		host_points[i].vz = vz;
		host_points[i].pointer = -1;
		//particles[k].type = j;
		double energy =  vx*vx + vy*vy + vz*vz;
		//fprintf(fe,"%d \t%lf \t%d \t%lf \t%lf \t%lf\n",i, energy, j,vx,vy,vz);
	}
}
void registerParticle(Particle &pt, sycl::queue& q, int prob_size, int working_size,std::string input_file,int j){
	//readHeader(pt, q);
    //readFile(pt, input_file, pt.params, q, j);
	std::cout << "Initialized Particles Randomly Successfully" << std::endl;
	std::cout << "Registering Particle with " << pt.END_NUM_Points << " particles." << std::endl;
	pt.NUM_Points = sycl::malloc_shared<int>(1, q);
    *(pt.NUM_Points) = 0;
    pt.capacity = pt.params.GRID_X * pt.params.GRID_Y * prob_size * working_size * 1.2;
    pt.prob_size = prob_size;
	pt.space = 5 * pt.END_NUM_Points;
	std::vector<Point> host_points(pt.END_NUM_Points);
	std::cout << "Initializing Particles Randomly" << std::endl;
	initParticlesRandomly(host_points,pt.END_NUM_Points,j);
	std::cout << "Initialized Particles Randomly Successfully" << std::endl;
	Point* d_points = sycl::malloc_device<Point>(pt.space, q);
	q.memcpy(d_points, host_points.data(), pt.END_NUM_Points * sizeof(Point)).wait();

	std::cout << "Copied Particles to Device Successfully" << std::endl;
	pt.d_points = d_points;
	if (!d_points) {
    	std::cerr << "Failed to allocate device memory for points\n";
    	std::exit(1);
	}
	std::cout << "Space allocated for particles: " << pt.space << " Number of Particles: " << pt.END_NUM_Points <<std::endl;
	// for(int i = 0;i < pt.END_NUM_Points;i++)
	// {
	// 	std::cout <<  i <<" " << d_points[i].x << " " << d_points[i].y << "\n";
	// }
    sycl_hashmap::SyclHashMap* hashmap = new sycl_hashmap::SyclHashMap(pt.capacity, q, prob_size*working_size, pt.params.GRID_X*pt.params.GRID_Y);
	std::cout << pt.params.GRID_X*pt.params.GRID_Y << " " << pt.capacity << " " << prob_size*working_size << "\n";
    pt.dmap = hashmap->device_view();
	std::cout << "Map Initialization" << std::endl;
    q.parallel_for<initKernel>(
        sycl::range<1>(pt.capacity),
        [=](sycl::id<1> item) {
            const size_t i = item[0];
            pt.dmap.keys[i] = -1;
            pt.dmap.values[i] = -1;
        }
    ).wait();
    q.parallel_for<initKeyKernel>(
        sycl::range<1>(pt.params.GRID_X*pt.params.GRID_Y),
        [=](sycl::id<1> item) {
            const size_t i = item[0];
            pt.dmap.key_idx[i] = 0;
            pt.dmap.delete_idx[i] = -1;
        }
    ).wait();
	std::cout << "Map Insertion Started" << std::endl;
	int gridSize = params1.GRID_X * params1.GRID_Y;
	std::cout << "DAT Params: " << params1.GRID_X << " " << params1.GRID_Y << " " << params1.dx << " " << params1.dy << "\n";
	//int TempFlag = 0;
    auto start = std::chrono::high_resolution_clock::now();
    q.submit([&](sycl::handler &h) {
        h.parallel_for<InsertKernel>(sycl::range<1>(pt.END_NUM_Points), [=](sycl::id<1> idx) {
            Point p = pt.d_points[idx];
            const double inv_dx = 1.0 / pt.params.dx;
            const double inv_dy = 1.0 / pt.params.dy;
            const int gx = static_cast<int>(p.x * inv_dx);
            const int gy = static_cast<int>(p.y * inv_dy);
            int grid_key = static_cast<size_t>(gy) * pt.params.GRID_X + gx;
            pt.d_points[idx].pointer = pt.dmap.insert(grid_key, static_cast<int>(idx[0]),pt.NUM_Points);
        });
    }).wait();
	//int TempCount = 0;
	// std::vector<int> countGridKeyValues(gridSize,0);
	// for(int i = 0;i < pt.END_NUM_Points;i++)
	// {
	// 	Point p = pt.d_points[i];
	// 		//if(pt.params.dx || pt.params.dy)return;
    //     const double inv_dx = 1.0 / pt.params.dx;
    // 	const double inv_dy = 1.0 / pt.params.dy;
    //     const int gx = static_cast<int>(p.x * inv_dx);
    //     const int gy = static_cast<int>(p.y * inv_dy);
    //     int grid_key = static_cast<size_t>(gy) * pt.params.GRID_X + gx;
	// 	countGridKeyValues[grid_key]++;
	// }
	// for(int i = 0;i < gridSize;i++)
	// {
	// 	std::cout << i << " " << countGridKeyValues[i] << "\n";
	// }
	//std::cout << " Out of Bound Particles =  " << TempCount << "\n";
    pt.final_mesh = sycl::malloc_shared<double>(static_cast<size_t>(pt.params.GRID_X) * pt.params.GRID_Y, q);
	pt.energy_mesh = sycl::malloc_shared<double>(static_cast<size_t>(pt.params.GRID_X) * pt.params.GRID_Y, q);
	pt.empty_space = sycl::malloc_shared<int>(pt.space,q);
	pt.empty_space_idx = sycl::malloc_device<int>(1,q);
	pt.new_additions = sycl::malloc_device<int>(1,q);

	pt.mover_left = sycl::malloc_shared<int>(pt.space,q);
	pt.mover_left_idx = sycl::malloc_shared<int>(1,q);
	q.memset(pt.new_additions,0,sizeof(int)).wait();
	q.memset(pt.empty_space_idx,0,sizeof(int)).wait();
	q.memset(pt.empty_space,-1,pt.END_NUM_Points*sizeof(int)).wait();
	
	std::cout << "Map Insertion Complete" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    tt += std::chrono::duration<double>(end - start).count();
}
void initVariables_legacy(GridParams& params, std::vector<Particle>& arr, sycl::queue& q){
    int gridSize = params.GRID_X * params.GRID_Y;
    //CtoM = sycl::malloc_shared<double>(num_type, q);
    corner_mesh = sycl::malloc_device<double>(4 * params.GRID_X * params.GRID_Y, q);
    // Creation of Temporary Arrays/Structures on the Device
    accessor_s1 = sycl::malloc_device<double>(static_cast<int>(prob_size*gridSize), q);
    accessor_s2 = sycl::malloc_device<double>(static_cast<int>(prob_size*gridSize), q);
    accessor_s3 = sycl::malloc_device<double>(static_cast<int>(prob_size*gridSize), q);
    accessor_s4 = sycl::malloc_device<double>(static_cast<int>(prob_size*gridSize), q);
    end_idx = sycl::malloc_device<int>(static_cast<int>(gridSize), q);
    particles = sycl::malloc_shared<Particle>(num_type, q);
    q.memcpy(particles, arr.data(), num_type*sizeof(Particle));
}

void freeVariables(sycl::queue& q){
    sycl::free(corner_mesh, q);
    sycl::free(accessor_s1, q);
    sycl::free(accessor_s2, q);
    sycl::free(accessor_s3, q);
    sycl::free(accessor_s4, q);
    sycl::free(end_idx, q);
    //sycl::free(CtoM,q);
    sycl::free(particles,q);
}

/**initialize charge to mass ratio for given index*/
void initCtoM(int index, double charge, double mass){
	CtoM[index] = charge / mass;
}

/**initialize weighing factor for given index*/
void updateWeighingFactor(int index, int simulationParticleCount, double actualParticleCount){
	weighingFactor[index] = actualParticleCount * surfArea / ( simulationParticleCount * dx * dy * dx * dy );
}


int readPICDAT(char* picdatfile,sycl::queue& q){
	int i;
	
	std::printf("%s\n", picdatfile);
	const char* pfn = "./";
	char tempic[50];

        std::strcpy(tempic, pfn);
        std::strcat(tempic, picdatfile);

	std::printf("\n\n\n%s\n\n\n", tempic);

	std::FILE *fd = std::fopen(picdatfile,"r");
	// std::FILE* fpp = std::fopen("heat_analysis.DAT","rw+");
	// if(fpp == NULL) {
	// 	std::printf("File not found: heat_analysis.DAT\n");
	// 	std::exit(1);		//heat_analysis file not found
	// }
	if(fd == NULL) {
		std::printf("File not found: PIC/data/PICDAT.DAT\n");
		std::exit(1);		//PICDAT file not found
	}

	std::cout << "File opened successfully" << std::endl;
	char useless1[8000];
	char useless2[8000];
	char temp;
	std::cout << "Reading PICDAT file" << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n]", useless1 , useless2) < 2) {
		std::printf("Inappropriate file Format1: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	std::cout << "1" << std::endl;
	if(std::fscanf(fd,"%d %c",&TIMESTEPS,&temp) < 2){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading TIMESTEPS)\n");
		std::exit(2);
	}
	std:: cout << "TIMESTEPS: " << TIMESTEPS << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n]", useless1 , useless2) < 2) {
		std::printf("Inappropriate file Format2: PIC/data/PICDAT.DAT \n");
		std::exit(2);
	}

	int NX,NY;
	if(std::fscanf(fd,"%lf %lf %d %d %c",&GRID_WIDTH,&GRID_HEIGHT,&NX,&NY,&temp) < 4){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading GRID_WIDTH, GRID_HEIGHT, NX, NY)\n");
		std::exit(2);
	}
	params1.NX = NX;
	params1.NY = NY;
	GRID_X = NX+1;
	GRID_Y = NY+1;
	dx = GRID_WIDTH/(NX);	
	dy = GRID_HEIGHT/(NY);

	params1.GRID_X = GRID_X;
	params1.GRID_Y = GRID_Y;
	params1.dx = dx;
	params1.dy = dy;
	params1.GRID_WIDTH = GRID_WIDTH;
	params1.GRID_HEIGHT = GRID_HEIGHT;

	std:: cout << "GRID_X: " << GRID_X << " GRID_Y: " << GRID_Y << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n]", useless1 , useless2) < 2) {
		std::printf("Inappropriate file Format3: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	if(std::fscanf(fd,"%lf %c",&dt,&temp) < 2){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading dt)\n");
		std::exit(2);
	}
	std::cout << "dt: " << dt << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format4: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	if(std::fscanf(fd,"%lf %lf %lf %c",&B0,&sigma,&mean,&temp) < 4){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading B0, sigma, mean, temp)\n");
		std::exit(2);
	}

	B0=B0/1000;
	mean=mean*params1.GRID_WIDTH;
	sigma=sigma/4;
	std::cout << "B0: " << B0 << " sigma: " << sigma << " mean: " << mean << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format5: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	if(std::fscanf(fd,"%lf %lf %lf %c",&electronTemp,&ionTemp,&gasTemp,&temp) < 4){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading electronTemp, ionTemp, gasTemp, temp)\n");
		std::exit(2);
	}
	std::cout << "electronTemp: " << electronTemp << " ionTemp: " << ionTemp << " gasTemp: " << gasTemp << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format6: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	
	if(std::fscanf(fd,"%lf %c",&pressure,&temp) < 2){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading pressure)\n");
		std::exit(2);
	}
	std::cout << "pressure: " << pressure << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format7: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	
	if(std::fscanf(fd,"%d %c",&isUniformInit,&temp) < 2){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading Uniform Initialization Flag)\n");
		std::exit(2);
	}
	std::cout << "isUniformInit: " << isUniformInit << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format8: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	if(std::fscanf(fd,"%lf %lf %lf %lf %c",&initFractionLeft, &initFractionRight, &initFractionTop, &initFractionBottom, &temp) < 5){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading xx1 xx2 yy1 yy2)\n");
		std::exit(2);
	}
	std::cout << "initFractionLeft: " << initFractionLeft << " initFractionRight: " << initFractionRight << " initFractionTop: " << initFractionTop << " initFractionBottom: " << initFractionBottom << std::endl;
	if(isUniformInit) surfArea = params1.GRID_WIDTH * params1.GRID_HEIGHT;
	else surfArea = (initFractionRight - initFractionLeft)*params1.GRID_WIDTH * (initFractionBottom - initFractionTop)*params1.GRID_HEIGHT;

	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format9: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	if(std::fscanf(fd,"%d %c",&doReinjection, &temp) < 2){
		std::printf("Inappropriate file Format : PIC/data/PICDAT.DAT (error while reading reinjection flag)\n");
		std::exit(2);
	}
	std::cout << "doReinjection: " << doReinjection << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format 10: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	if(std::fscanf(fd,"%d %c",&scaleFactor, &temp) < 2){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading scale factor)\n");
		std::exit(2);
	}
	std::cout << "scaleFactor: " << scaleFactor << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format11: PIC/data/PICDAT.DAT \n");
		std::exit(2);
	}

	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format12: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}

	int initParticleFlag;
	if(std::fscanf(fd,"%d %c",&initParticleFlag,&temp) < 2){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading initParticleFlag)\n");
		std::exit(2);
	}
	std::cout << "initParticleFlag: " << initParticleFlag << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format13: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}

	if(std::fscanf(fd,"%lf %c",&epsilon,&temp) < 2){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading epsilon)\n");
		std::exit(2);
	}
	std::cout << "epsilon: " << epsilon << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format14: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}

	if(std::fscanf(fd,"%d %c",&periodicBoundary,&temp) < 2){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading periodic boundary flag)\n");
		std::exit(2);
	}
	std::cout << "periodicBoundary: " << periodicBoundary << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format15: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	
	if(std::fscanf(fd,"%[^\n] %c",gasName,&temp) < 1){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading gasname)\n");
		std::exit(2);
	}
	trim(gasName,30);
	std::cout << "gasName: " << gasName << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format16: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	
	if(std::fscanf(fd,"%lf %c",&attachmentFactor,&temp) < 2){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading attachmentFactor)\n");
		std::exit(2);
	}
	std::cout << "attachmentFactor: " << attachmentFactor << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format17: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	
	if(std::fscanf(fd,"%d %c",&collisionFlag,&temp) < 2){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading collision flag)\n");
		std::exit(2);
	}
	std::cout << "collisionFlag: " << collisionFlag << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format18: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	

//
        if(std::fscanf(fd,"%d %lf %lf %c",&powerMode, &heatFrequency, &power, &temp) < 4){
                std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading heating variables)\n");
                std::exit(2);
        }
		std::cout << "powerMode: " << powerMode << " heatFrequency: " << heatFrequency << " power: " << power << std::endl;
//---std::printf("heat variables: %d %lf %lf\n", powerMode, heatFrequency, power);

	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
                std::printf("Inappropriate file Format99: PIC/data/PICDAT.DAT\n");
                std::exit(2);
	}
//

	if(std::fscanf(fd,"%d %c",&printInterval,&temp) <= 1){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading printInterwal)\n");
		std::exit(2);
	}
	std::cout << "printInterval: " << printInterval << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
                std::printf("Inappropriate file Format99: PIC/data/PICDAT.DAT\n");
                std::exit(2);
	}
//

	if(std::fscanf(fd,"%d %d %c",&STARTprintspaceavg,&ENDprintspaceavg,&temp) < 2){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading start and end print spaceavg)\n");
		std::exit(2);
	}
	std::cout << "STARTprintspaceavg: " << STARTprintspaceavg << " ENDprintspaceavg: " << ENDprintspaceavg << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) < 2) {
		std::printf("Inappropriate file Format19: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	
	if(std::fscanf(fd,"%d %c",&num_type,&temp) <= 1){
		std::printf("Inappropriate file Format: PIC/data/PICDAT.DAT (error while reading no of species)\n");
		std::exit(2);
	}
	std:: cout << "num_type: " << num_type << std::endl;
	if(std::fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) <= 1) {
		std::printf("Inappropriate file Format20: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}
	if(std::fscanf(fd,"%[^\n] ", useless1) <= 0) {
		std::printf("Inappropriate file Format21: PIC/data/PICDAT.DAT\n");
		std::exit(2);
	}

	//setting up the variables required for initializing particle array
	//determines how much particle each rank will handle.


	CtoM = sycl::malloc_shared<double>(num_type, q);
	weighingFactor = sycl::malloc_shared<double>(num_type, q);
	simulationParticles = sycl::malloc_shared<int>(num_type, q);
	initial_simulation_particles = sycl::malloc_shared<long>(num_type, q);
	actualParticles = sycl::malloc_shared<double>(num_type, q);
	particleCharges = sycl::malloc_shared<double>(num_type, q);
	// particleStartIndex = (int *) std::calloc(totalParticleTypes , sizeof(int));
	PARTICLES = 0;
	// PARTICLES_BUFFERSIZE = 0;



	for(i=0;i<num_type;i++){
		
		int index;
		double simulation,actual,charge,mass;
		
		if(std::fscanf(fd,"%d %lf %lf %lf %lf",&index,&simulation,&actual,&charge,&mass) <= 4){
			std::printf("Inappropriate file Format10: PIC/data/PICDAT.DAT (error while reading tabular data)\n");
			std::exit(2);
		}
		
		initCtoM(i,charge,mass);
		updateWeighingFactor(i,simulation,actual);
		
		// PARTICLES += simulation; Update for MPI as below
		
		PARTICLES += simulation;
		simulationParticles[i] = (int) simulation;
		actualParticles[i] = actual;
		particleCharges[i] = charge;
		
		// int shareOfParticles = ((int) simulation) / MPI_WORLD_SIZE;
		// if (mpi_world_rank == MPI_WORLD_SIZE - 1) {
		// 	shareOfParticles += (((int) simulation) % MPI_WORLD_SIZE);
		// }
		// PARTICLES += shareOfParticles;
		// simulationParticles[i] = shareOfParticles;
		// initial_simulation_particles[i] = (long) simulationParticles[i];
		// actualParticles[i] = actual; // Not updated for MPI
		// particleCharges[i] = charge;
		
	
	}
	//PARTICLES_BUFFERSIZE += totalParticleTypes * ((int)(PARTICLES * (1+auxiliarySpaceFraction)/totalParticleTypes) + 1);

	// particleStartIndex[0] = 0;
	// for(i=1 ;i<totalParticleTypes ;i++){
	// 	particleStartIndex[i] = particleStartIndex[i-1] + simulationParticles[i-1] + ((int)(PARTICLES * (auxiliarySpaceFraction)/totalParticleTypes) + 1);
	// 	//std::printf("species : %d startIndex: %d\n",i,particleStartIndex[i] );
	// }

	std::fclose(fd);
	return initParticleFlag;

}


void readGeometry(sycl::queue& q){

	FILE *fd = fopen("./geometry.conf","r");
	if(fd == NULL) {
		printf("File not found: PIC/data/geometry.conf\n");
		exit(1);		//geometry file not found
	}
	char useless1[150];
	char useless2[150];
	char temp;

	if(fscanf(fd,"%[^\n] %[^\n] %d %c", useless1 , useless2 , &geometryCount , &temp) <= 3) {
		printf("Inappropriate file Format: PIC/data/geometry.conf\n");
		exit(2);
	}
	if(periodicBoundary == 1 && geometryCount > 3) {
		printf("Inconsistant input files: Periodic boundary expects geometry count = 3\n");
		exit(9);	
	}
	//printf("%d\n\n\n\n\n\n",geometryCount);
	if(fscanf(fd,"%[^\n] %[^\n] ", useless1 , useless2 ) <= 1) {
		printf("Inappropriate file Format: PIC/data/geometry.conf\n");
		exit(2);
	}

	geometries = sycl::malloc_shared<Geometry>(geometryCount, q);

	if(geometries == NULL) {
		printf("Couldn't allocate memory: geometry array could not be created \n");
		exit(3);
	}

	int i=geometryCount;
	while(i > 0){

		double x1,y1,x2,y2,pot;
		if(fscanf(fd , "%lf %lf %lf %lf %lf", &x1 , &y1 , &x2 , &y2 , &pot) <= 4){
			printf("Inappropriate file Format: PIC/data/geometry.conf\n");
			exit(2);
		}

		geometries[geometryCount-i].x1 = (int) (params1.GRID_X-1) * x1;
		geometries[geometryCount-i].y1 = (int) (params1.GRID_Y-1) * y1;
		geometries[geometryCount-i].x2 = (int) (params1.GRID_X-1) * x2;
		geometries[geometryCount-i].y2 = (int) (params1.GRID_Y-1) * y2;
		geometries[geometryCount-i].potential = pot;

		i--;
	}
	fclose(fd);
}


/** Initialize magnetic field as Gaussian with 'width', 'centre' and 'max' defined in the PICDAT */
void initMagneticField_Benchmark(double a, double mean, double std){
	int i,j;
         for (i=0; i < GRID_X; i++){
              if (i < ibmax){
                magneticField[i] = (0.005699 + (a1 * exp(-pow(((i - ibmax)*dx)/sigma1, 2)/2)) + b1);
                }
              else
               {
                 magneticField[i] =(0.000015 + (a2 * exp(-pow(((i - ibmax)* dx)/sigma1, 2)/2)) + b2);
               }
            }

	for(i = 1; i < GRID_Y ; i++){
		memcpy(magneticField+i*GRID_X, magneticField, GRID_X*sizeof(double));
	}
}


/** Initialize magnetic field as Gaussian with 'width', 'centre' and 'max' defined in the PICDAT */
void initMagneticField(double a, double mean, double std){
	int i,j;
	for(i=0 ; i < GRID_X ; i++){
		//magneticField[i] = a * exp( -  pow( (i*dx - mean) , 2 ) / ( 2 * pow( std , 2 )  ) );
		magneticField[i] = a;
	}

	for(i = 1; i < GRID_Y ; i++){
		memcpy(magneticField+i*GRID_X, magneticField, GRID_X*sizeof(double));
	}
}


void initVariables(char* PIC_File_Name, sycl::queue& q,std::vector<std::string>& input_files){
	std::cout << "Reading PICDAT File" << std::endl;
	// Particle and Grid Variable Initialization
	int initParticleFlag = readPICDAT(PIC_File_Name, q);
	std::cout << "Read PICDAT File Successfully" << std::endl;
	std::cout << "Initializing Variables" << std::endl;
	rho = sycl::malloc_shared<double>(params1.GRID_X * params1.GRID_Y,q);//(double *) calloc(GRID_X * GRID_Y , sizeof(double));
	phi = sycl::malloc_shared<double>(params1.GRID_X * params1.GRID_Y,q);//(double *) calloc(GRID_X * GRID_Y , sizeof(double));
	TPENsum = sycl::malloc_shared<double>(num_type,q);//(double *) calloc(totalParticleTypes , sizeof(double));

	//CtoM = sycl::malloc_shared<double>(num_type,q);//(double *) calloc(totalParticleTypes , sizeof(double));
	magneticField =  sycl::malloc_shared<double>(params1.GRID_X * params1.GRID_Y,q); //(double *) calloc (GRID_X * GRID_Y , sizeof(double));
	electricField = sycl::malloc_shared<Field>(params1.GRID_X * params1.GRID_Y,q); //(Field *) calloc (GRID_X * GRID_Y , sizeof(Field));
	//energy =sycl::malloc_shared<double>(num_type * params1.GRID_X * params1.GRID_Y,q); //(double *) calloc (totalParticleTypes*GRID_X * GRID_Y , sizeof(double));
	//numberDensity = (double *) calloc(totalParticleTypes*GRID_X*GRID_Y, sizeof(double));

	energy1 = sycl::malloc_shared<double>(num_type * params1.GRID_X * params1.GRID_Y,q);//(double *) calloc(totalParticleTypes*GRID_X * GRID_Y, sizeof(double));
	//numberDensity1 = (double *) calloc(totalParticleTypes * GRID_X * GRID_Y, sizeof(double));
	rho1 = sycl::malloc_shared<double>(params1.GRID_X * params1.GRID_Y,q); //(double *) calloc(GRID_X * GRID_Y, sizeof(double));
	TPENsum1 = sycl::malloc_shared<double>(num_type,q);//(double *) calloc(totalParticleTypes, sizeof(double));
	

	//particles = (Particle *) calloc (PARTICLES_BUFFERSIZE , sizeof(Particle));

	eledensavg = sycl::malloc_shared<double>(params1.GRID_X,q);//(double *) calloc(GRID_X, sizeof(double)); ///M---average ele density in space-time avg
	iondensavg = sycl::malloc_shared<double>(params1.GRID_X,q);//(double *) calloc(GRID_X, sizeof(double)); ///M---average ele density in space-time avg
	Teavg = sycl::malloc_shared<double>(params1.GRID_X,q);//(double *) calloc(GRID_X, sizeof(double));  	///M---average ele temp in space-time average
	phiavg = sycl::malloc_shared<double>(params1.GRID_X,q);//(double *) calloc(GRID_X, sizeof(double)); 	///M---average potential in space-time average
	Tiavg = sycl::malloc_shared<double>(params1.GRID_X,q);//(double *) calloc(GRID_X, sizeof(double));		///M---average ion temp in space-time average


	corner_mesh = sycl::malloc_device<double>(4 * params1.GRID_X * params1.GRID_Y, q);
	corner_mesh_enrgy = sycl::malloc_device<double>(4 * params1.GRID_X * params1.GRID_Y, q);
    // Creation of Temporary Arrays/Structures on the Device
    accessor_s1 = sycl::malloc_device<double>(static_cast<int>(prob_size*params1.GRID_X * params1.GRID_Y), q);
    accessor_s2 = sycl::malloc_device<double>(static_cast<int>(prob_size*params1.GRID_X * params1.GRID_Y), q);
    accessor_s3 = sycl::malloc_device<double>(static_cast<int>(prob_size*params1.GRID_X * params1.GRID_Y), q);
    accessor_s4 = sycl::malloc_device<double>(static_cast<int>(prob_size*params1.GRID_X * params1.GRID_Y), q);
	accessor_s1_enrgy = sycl::malloc_device<double>(static_cast<int>(prob_size*params1.GRID_X * params1.GRID_Y), q);
    accessor_s2_enrgy = sycl::malloc_device<double>(static_cast<int>(prob_size*params1.GRID_X * params1.GRID_Y), q);
    accessor_s3_enrgy = sycl::malloc_device<double>(static_cast<int>(prob_size*params1.GRID_X * params1.GRID_Y), q);
    accessor_s4_enrgy = sycl::malloc_device<double>(static_cast<int>(prob_size*params1.GRID_X * params1.GRID_Y), q);
    end_idx = sycl::malloc_device<int>(static_cast<int>(params1.GRID_X * params1.GRID_Y), q);
	deleted_mask = sycl::malloc_shared<int>(simulationParticles[0], q);
	deleted_offset = sycl::malloc_shared<int>(simulationParticles[0], q);
	back_invalid_mask = sycl::malloc_shared<int>(simulationParticles[0], q);
	back_offset = sycl::malloc_shared<int>(simulationParticles[0], q);
	// empty_space = sycl::malloc_device<int>(simulationParticles[0], q);
	// empty_space_idx = sycl::malloc_device<int>(1, q);
	
	// q.memset(empty_space_idx, 0, sizeof(int)).wait();
	// q.memset(empty_space, -1, sizeof(int)*simulationParticles[0]).wait();
	std::cout << "Initialized Variables Successfully" << std::endl;
	initUtility();
	std::cout << "Initializing and Storing Particles" << std::endl;
	// Particle Initialization and Data Structure Storing
	std::vector<Particle> arr(num_type);
	for(int i=0 ; i < num_type ; i++){
		//arr[i].input_file = (char *) std::calloc(30 , sizeof(char));
		arr[i].charge_sign = particleCharges[i];
		//arr[i].prob_size = prob_size;
		arr[i].END_NUM_Points = simulationParticles[i];
		//cout << "Species: " << i << " Number of Simulation Particles: " << arr[i].END_NUM_Points << endl;
		arr[i].params = params1;
		std::cout << "Species: " << i << " Number of Simulation Particles: " << arr[i].END_NUM_Points << " Now creating and storing Particles in DS"<< std::endl;
		registerParticle(arr[i], q, prob_size, working_size, input_files[i],i);
	}
    particles = sycl::malloc_shared<Particle>(num_type, q);
    q.memcpy(particles, arr.data(), num_type*sizeof(Particle));
	std::cout << "Initialized and Stored Particles Successfully" << std::endl;
	// error checking
	if(rho == NULL || phi == NULL || CtoM == NULL || magneticField == NULL || electricField == NULL || magneticField == NULL || particles == NULL){
		printf("Couldn't allocate memory: rho | phi | magneticField | electricField | particles");
		exit(3);
	}
	dng = pressure /(boltzmannConstant * gasTemp);
	readGeometry(q);
	std::cout << "Read Geometry File Successfully" << std::endl;
	//initUtility();
	//initParticles(particles,initParticleFlag);
	initMagneticField(B0,mean,sigma);
	std::cout << "Initialized Magnetic Field Successfully" << std::endl;
	std::cout << "Initialization Complete" << std::endl;
}