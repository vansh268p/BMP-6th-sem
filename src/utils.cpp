#include "utils.hpp"
#include <sycl/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/mkl/rng/device.hpp>
#include "sycl_hashmap.hpp"
#include <cstring>
#include <time.h>
#include <mkl.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <chrono>

#define PI 3.141592654
namespace rngd = oneapi::mkl::rng::device;
double seed;
size_t trim(char *out, size_t len)
{
	int i=len-2;
	while(out[i--] == 32);
	out[i+2] = 0;

	return i+3;
}


void cleanup_map(const GridParams& params,
                 sycl::queue& q,
                 sycl_hashmap::DeviceView& dmap,
                 Point* points_device)
{
    int global_size = params.GRID_X * params.GRID_Y;
    q.parallel_for<CleanUpKernel>(
        sycl::range<1>(global_size),
        [=](sycl::id<1> item) {
            const size_t i = item[0];
            dmap.cleanup(i,points_device);
			dmap.cleanup_clear_freelist(i,points_device);
        }
    ).wait();
}

void initUtility(){
	std::srand(time(NULL));
    seed = (std::rand()/(double)RAND_MAX)*123456789;
}

double ran2(){
	
	double r = (std::rand()/((double) (RAND_MAX) + 1));
	if(r < 1e-10) r = 1e-10;
	return r;
}

void vmaxwn2(double *vx, double *vy, double *vz, double v){
	double z1 = sqrt(-log(ran2()));
	double z2 = 6.283185307 * ran2();
	*vx = v * z1 * cos(z2);
	*vy = v * z1 * sin(z2);
	if(stored == 0){
		z1 = sqrt(-log(ran2()));
		z2 = 6.283185307 * ran2();
		*vz = v * z1 * cos(z2);
		stored = z1 * sin(z2);
	}
	else{
		*vz = v * stored;
		stored = 0;
	}
}


// returns total kinetic energy as double
double calculate_energy(sycl::queue &q,
                        const Point *points_device, // device/USM pointer to Point array
                        int END_NUM_Points,
                        const double *CtoM,         // device-accessible or host-captured array
                        int ptype)
{
    if (END_NUM_Points <= 0) return 0.0;

    // Precompute denominator (q/m absolute) and capture it by value
    const double denom = std::fabs(CtoM[ptype]);
    if (denom == 0.0) return 0.0;

    double total_energy = 0.0;
    {
        // host-visible buffer that reduction will write into
        sycl::buffer<double, 1> result_buf(&total_energy, sycl::range<1>(1));

        q.submit([&](sycl::handler &h) {
            // create reduction accessor on the buffer
            auto red = sycl::reduction(result_buf, h, std::plus<double>());

            h.parallel_for<EnergyKernel>(
                sycl::range<1>(static_cast<size_t>(END_NUM_Points)), red,
                [=](sycl::id<1> idx, auto &sum) {
                    const size_t i = idx[0];
                    const Point &p = points_device[i];
                    // skip inactive particles if you mark pointer == -1
                    if (p.pointer == -1) return;

                    double pvx = p.vx;
                    double pvy = p.vy;
                    double pvz = p.vz;

                    double v2 = pvx * pvx + pvy * pvy + pvz * pvz;
                    double e = 0.5 * v2 / denom;
                    sum += e;
                });
        }).wait(); // wait for kernel and buffer writeback
    } // result_buf destructor ensures value copied back to total_energy

    return total_energy;
}

int calculate_prob_size(sycl::queue &q,
							sycl_hashmap::DeviceView& dmap,
							int max_prob_size,
							int probing_length,
							const GridParams& params
						)
{
	int max_value = 0;
    {
        // host-visible buffer that reduction will write into
        sycl::buffer<int, 1> result_buf(&max_value, sycl::range<1>(1));
		int gridSize = params.GRID_X * params.GRID_Y;
        q.submit([&](sycl::handler &h) {
            // create reduction accessor on the buffer
            auto red = sycl::reduction(result_buf, h, sycl::maximum<int>());

            h.parallel_for<FindLengthKernel>(
                sycl::range<1>(static_cast<size_t>(gridSize)), red,
                [=](sycl::id<1> idx, auto &maxv) {
                    const size_t i = idx[0];
                    int k_idx = dmap.key_idx[i];
                    maxv.combine(k_idx);
                });
        }).wait(); // wait for kernel and buffer writeback
    } // result_buf destructor ensures value copied back to total_energy
	int current_prob_size = 8;
	int current_working_size = std::ceil((double)probing_length/current_prob_size);
	int diff = abs(current_prob_size - current_working_size);
	int best_prob_size = current_prob_size;
	for(int i = current_prob_size; i <= max_prob_size;i += 8)
	{
		current_working_size = std::ceil((double)max_value/i);
		if(abs(i - current_working_size) <= diff)
		{
			best_prob_size = i;
			diff = abs(i - current_working_size);
		}
	}
	return best_prob_size;
}

double calc_energy_serial(const Point *points_host, // host pointer to Point array
						  int END_NUM_Points,
						  const double *CtoM,     // host pointer to CtoM array
						  int ptype)
{
	if (END_NUM_Points <= 0) return 0.0;

	// Precompute denominator (q/m absolute) and capture it by value
	const double denom = std::fabs(CtoM[ptype]);
	if (denom == 0.0) return 0.0;

	double total_energy = 0.0;
	
	for (int i = 0; i < END_NUM_Points; i++) {
		const Point &p = points_host[i];
		// skip inactive particles if you mark pointer == -1
		if (p.pointer == -1) continue;

		double pvx = p.vx;
		double pvy = p.vy;
		double pvz = p.vz;

		double v2 = pvx * pvx + pvy * pvy + pvz * pvz;
		double e = 0.5 * v2 / denom;
		total_energy += e;
	}

	return total_energy;
}

// void printEnergy(char *dir, int iteration) {

// 	int i,j;
// 	char fileName[296] = "",fileName1[296]="", tmp[100] = "";
// 	//---Print electron energy in eV ------//
// 	strcat(fileName, dir);
// 	strcat(fileName, "/electronenergy/");  //electron energy
// 	sprintf(tmp, "%d", iteration+Iter);
// 	strcat(fileName, tmp);
// 	FILE *fd=fopen(fileName,"w");
// 	if(fd == NULL){
// 		printf("File not Created: PIC/out/<DIR_NAME>/electronenergy.out\n");
// 		exit(1);
// 	}

// 	for(i=0;i<GRID_Y;i++){
// 		for(j=0;j<GRID_X - 1;j++){
// 			fprintf(fd,"%lf ",0.66667*energy[0*GRID_X*GRID_Y +i*GRID_X+j]/(numberDensity[0*GRID_X*GRID_Y + i*GRID_X+j]+1.0));
// 		}
// 		fprintf(fd,"%lf\n",0.66667*energy[0*GRID_X*GRID_Y +i*GRID_X+j]/(numberDensity[0*GRID_X*GRID_Y + i*GRID_X+j]+1.0));
// 	}
// 	fclose(fd);

// 	//-----print ion energy in eV---------//

// 	strcat(fileName1, dir);
// 	strcat(fileName1, "/ionenergy/");  //electron energy
// 	sprintf(tmp, "%d", iteration+Iter);
// 	strcat(fileName1, tmp);
// 	FILE *fe=fopen(fileName1,"w");
// 	if(fe == NULL){
// 		printf("File not Created: PIC/out/<DIR_NAME>/ionenergy.out\n");
// 		exit(1);
// 	}

// 	for(i=0;i<GRID_Y;i++){
// 		for(j=0;j<GRID_X - 1;j++){
// 			fprintf(fe,"%lf ",0.66667*energy[1*GRID_X*GRID_Y +i*GRID_X+j]/(numberDensity[1*GRID_X*GRID_Y + i*GRID_X+j]+1.0));
// 		}
// 		fprintf(fe,"%lf\n",0.66667*energy[1*GRID_X*GRID_Y +i*GRID_X+j]/(numberDensity[1*GRID_X*GRID_Y + i*GRID_X+j]+1.0));
// 	}
// 	fclose(fe);

// }

// int store_deleted_indices(int index,int* empty_space,int* empty_space_count,int n){
// 	auto atomic_count = sycl::atomic_ref<int,
// 		sycl::memory_order::relaxed,
// 		sycl::memory_scope::device,
// 		sycl::access::address_space::global_space>( *empty_space_count );
// 	int k_idx = atomic_count.fetch_add(1);
// 	if(k_idx <= n){
// 		empty_space[k_idx - 1] = index;
// 		return 1;
// 	}
// 	return 0;
// }

int compact_particles(Point* points_device,
					   sycl_hashmap::DeviceView& dmap,
					   sycl::queue& q,
					   int* empty_space,
					   int* empty_space_idx,
					   int END_NUM_Points,
					   int* NUM_Points,
					   double& alloc_time,
					   double& scan_time)
{
	//Number of Deleted Particles
	int num_deleted = 0;
	q.memcpy(&num_deleted, empty_space_idx, sizeof(int)).wait();

	// Initial check
	if(num_deleted == 0) return END_NUM_Points; // No deleted particles to compact

	// Update NUM_Points
	*NUM_Points = END_NUM_Points - num_deleted;
	//std::cout << "Compacting " << num_deleted << " particles. New count: " << *NUM_Points << std::endl;

	// Allocate masks and offsets
	auto start_alloc = std::chrono::high_resolution_clock::now();
	int* deleted_mask = sycl::malloc_shared<int>(num_deleted, q);
	int* back_valid_mask = sycl::malloc_shared<int>(num_deleted, q);
	int* back_offset = sycl::malloc_shared<int>(num_deleted, q);
	int* deleted_offset = sycl::malloc_shared<int>(num_deleted, q);
	auto end_alloc = std::chrono::high_resolution_clock::now();
	alloc_time += std::chrono::duration<double>(end_alloc - start_alloc).count();
	// for(int i = 0;i < std::min(num_deleted,100);i++)
	// {
	// 	std::cout << "Empty Space[" << i << "] = " << empty_space[i] << " - "<<  (empty_space[i] < END_NUM_Points) <<"\n";
	// }
	// Mask Kernel
	q.submit([&](sycl::handler &h) {
		h.parallel_for<MaskKernel>(
			sycl::range<1>(static_cast<size_t>(num_deleted)),
			[=](sycl::id<1> idx) {
				deleted_mask[idx] = (empty_space[idx] < *NUM_Points) ? 1 : 0;
				back_valid_mask[idx] = (points_device[END_NUM_Points - idx - 1].pointer != -1) ? 1 : 0;
			}
		);
	}).wait();
	//std::cout << "Masks Computed, Performing Scans" << std::endl;

	// Scan to get offsets as per device policy
	auto start_scan = std::chrono::high_resolution_clock::now();
	auto policy = oneapi::dpl::execution::make_device_policy(q);
	oneapi::dpl::exclusive_scan(policy, deleted_mask, deleted_mask + num_deleted, deleted_offset,0);
	oneapi::dpl::exclusive_scan(policy, back_valid_mask, back_valid_mask + num_deleted, back_offset,0);
	q.wait();
	auto end_scan = std::chrono::high_resolution_clock::now();
	scan_time += std::chrono::duration<double>(end_scan - start_scan).count();
	//int total_back_valid = 0;
	int back_offset_last = back_offset[num_deleted - 1];
	int back_valid_last = back_valid_mask[num_deleted - 1];
	int total_back_valid = back_offset_last + back_valid_last;


	//Checking total back valid
	int deleted_offset_last = deleted_offset[num_deleted - 1];
	int deleted_last = deleted_mask[num_deleted - 1];
	int delete_needed = deleted_offset_last + deleted_last;
	if(delete_needed != total_back_valid){

		std::cerr << "Error in Scan -- Mismatch! - (" << delete_needed << ", " << total_back_valid << ")" << std::endl;
		exit(1);
	}

	// Scatter Kernel
	start_alloc = std::chrono::high_resolution_clock::now();
	int* deleted_pos = sycl::malloc_shared<int>(total_back_valid, q);
	int* back_pos = sycl::malloc_shared<int>(total_back_valid, q);
	end_alloc = std::chrono::high_resolution_clock::now();
	alloc_time += std::chrono::duration<double>(end_alloc - start_alloc).count();
	//std::cout << "Starting Scatter Kernel" << std::endl;
	q.submit([&](sycl::handler &h) {
		h.parallel_for<ScatterKernel>(
			sycl::range<1>(static_cast<size_t>(num_deleted)),
			[=](sycl::id<1> idx) {
				if(deleted_mask[idx] == 1){
					int pos = deleted_offset[idx];
					deleted_pos[pos] = empty_space[idx];
				}
				if(back_valid_mask[idx] == 1){
					int pos = back_offset[idx];
					back_pos[pos] = END_NUM_Points - idx - 1;
				}
			}
		);
	}).wait();
	//std::cout << "Total Back Valid: " << total_back_valid << std::endl;

	// Compaction Kernel
	//std::cout << "Starting Compaction Kernel" << std::endl;
	q.submit([&](sycl::handler &h) {
		h.parallel_for<CompactionKernel>(
			sycl::range<1>(static_cast<size_t>(total_back_valid)),
			[=](sycl::id<1> idx) {
				int from_idx = back_pos[idx];
				int to_idx = deleted_pos[idx];
				points_device[to_idx] = points_device[from_idx];
				if(points_device[to_idx].pointer != -1)
					dmap.values[points_device[to_idx].pointer] = to_idx;
				points_device[from_idx].pointer = -1;
			}
		);
	}).wait();
	// New END_NUM_Points
	int END_NUM_Points_new = *NUM_Points;

	// Free allocated memory
	sycl::free(deleted_mask, q);
	sycl::free(back_valid_mask, q);
	sycl::free(back_offset, q);
	sycl::free(deleted_offset, q);
	sycl::free(deleted_pos, q);
	sycl::free(back_pos, q);
	
	// Reset empty_space_idx
	q.memset(empty_space_idx, 0, sizeof(int)).wait();
	//std::cout << "Compaction Completed" << std::endl;
	return END_NUM_Points_new;
}

int generate_new_electrons(Point* point_device,
                           sycl::queue& q,
                           const GridParams& params,
                           sycl_hashmap::DeviceView& dmap,
                           int END_NUM_Points,    // host-side value (int)
                           int* NUM_Points,       // USM-shared pointer to device int (do NOT modify before kernel)
                           int space,             // capacity on device (host int)
                           double* CtoM,
                           int electronTemp,
                           double& stored,
                           int num_new)
{
    if (num_new == 0) return END_NUM_Points;

    // safety: ensure we have room
    if (END_NUM_Points + num_new > space) {
        std::cerr << "Error: Exceeding allocated particle space while generating new electrons!"
                  << " END_NUM_Points=" << END_NUM_Points
                  << " num_new=" << num_new << " space=" << space << std::endl;
        return END_NUM_Points; // or exit depending on desired behavior
    }

    // Host generate points into a host vector (you already do this)
    std::vector<Point> h_points;
    h_points.reserve(num_new);
    for (int i = 0; i < num_new; ++i) {
        double ran3 = ((double)std::rand()) / ((double)RAND_MAX);
        double x = 0.024;
        double y = params.GRID_HEIGHT * ran3;

        double vx, vy, vz;
        double v = sqrt(2.0 * fabs(CtoM[0]) * electronTemp);
        double z1 = sqrt(-log(ran2()));
        double z2 = 6.283185307 * ran2();
		//std::cout << ran3 << " " << z1 << " " << z2 << std::endl;
        vx = v * z1 * cos(z2);
        vy = v * z1 * sin(z2);
        if (stored == 0) {
            z1 = sqrt(-log(ran2()));
            z2 = 6.283185307 * ran2();
            vz = v * z1 * cos(z2);
            stored = z1 * sin(z2);
        } else {
            vz = v * stored;
            stored = 0;
        }
        Point p;
        p.x = x; p.y = y; p.z = 0.0;
        p.vx = vx; p.vy = vy; p.vz = vz;
        p.pointer = -1;
        h_points.push_back(p);
    }

    // copy to device contiguous block
    int start_idx = END_NUM_Points;
    size_t copy_bytes = static_cast<size_t>(num_new) * sizeof(Point);
    q.memcpy(&point_device[start_idx], h_points.data(), copy_bytes).wait();

    // Launch kernel to insert into hashmap. Do NOT modify *NUM_Points on host before this.
    // Pass only device pointers and small PODs into kernel: point_device, start_idx, space, NUM_Points, params.GRID_X/Y, dx/dy.
    const int gridX = params.GRID_X;
    const int gridY = params.GRID_Y;
    const double inv_dx = 1.0 / params.dx;
    const double inv_dy = 1.0 / params.dy;
    const int total_grid_cells = gridX * gridY;

	int *error_flag = sycl::malloc_shared<int>(1, q);
	*error_flag = 0;

    q.submit([&](sycl::handler& h) {
        h.parallel_for<class newElectronKernel_safe>( sycl::range<1>((size_t)num_new),
        [=](sycl::id<1> idx) {
			if(*error_flag != 0) return; // early out on error
            int local = static_cast<int>(idx[0]);
            int global_idx = start_idx + local; // target index in point_device

            // bounds check to be safe (shouldn't happen due to earlier check)
            if (global_idx < 0 || global_idx >= space) return;

            Point p = point_device[global_idx]; // local copy

            int gx = static_cast<int>(p.x * inv_dx);
            int gy = static_cast<int>(p.y * inv_dy);

            // validate grid indices
            if (gx < 0 || gx >= gridX || gy < 0 || gy >= gridY) {
                // invalid grid -> do not insert; mark pointer -1 so compaction can pick it up
                point_device[global_idx].pointer = -1;
                return;
            }

            int grid_key = gy * gridX + gx;

            // Insert into hashmap. dmap.insert should be safe and expects device-accessible NUM_Points.
            int new_ptr = dmap.insert(grid_key, global_idx, NUM_Points);
            // dmap.insert may return -1 on failure. Handle it.
            if (new_ptr < 0) {
                // insertion failed; mark pointer invalid and return
                point_device[global_idx].pointer = -1;
				*error_flag = 1; // set error flag
            } else {
                // store pointer returned by hashmap
                point_device[global_idx].pointer = new_ptr;
            }
        });
    }).wait(); // ensure kernel completes before host reads NUM_Points or runs compaction
	if(*error_flag != 0) {
		std::cerr << "Error: Hashmap insertion failed during new electron generation." << std::endl;
		sycl::free(error_flag, q);
		std::exit(1);
	}
    // After successful insertions we now update host-side counters:
    // We must read the device-side NUM_Points (if you need it) or simply increase END_NUM_Points by num_new.
    // IMPORTANT: do NOT assume dmap.insert changed *NUM_Points in a particular way; many implementations do per-cell insertion not global count.
    // If your program logic requires *NUM_Points to be the total count on device, update it here:
    //       *NUM_Points_device = read from the device if necessary (e.g., q.memcpy(&host_int, NUM_Points, ...).wait();)
    // For simplicity we update host END_NUM_Points and leave device NUM_Points as-is or you can sync it:
    int new_END = END_NUM_Points + num_new;

    // Optionally synchronize NUM_Points device value back to host integer if you need it:
    // int device_num_points = 0;
    // q.memcpy(&device_num_points, NUM_Points, sizeof(int)).wait();
    // then set *NUM_Points_host accordingly or verify consistency.

    return new_END;
}

// void generate_pairs(Particle& electron, Particle& ion, sycl::queue& q, const GridParams& params){
// 	// Placeholder for pair generation logic
// 	// This function can be expanded based on specific pairing criteria
// 	// For now, it does nothing
// 	std::cout << "Generating new electron-ion pairs." << std::endl;
// 	int i=0,j;
// 	double x,y, ran3, ran4,fn;
// 	int lower_limit = 0;
// 	int upper_limit = 1;
// 	ionimax = 5.23e23;
// 	fn = params.GRID_HEIGHT * dt * (2.0/PI) * (x22 - x11)  * ionimax *(1.0/(1.67e6));
// 	int electron_start = electron.END_NUM_Points;
// 	int ion_start = ion.END_NUM_Points;
// 	std::vector<Point> h_electrons(static_cast<size_t>(fn));
// 	std::vector<Point> h_ions(static_cast<size_t>(fn));
// 	std::cout << "Generation started, Number of pairs to generate: " << fn << std::endl;
// 	//int num_new = 0;
// 	for(j=0;j <= fn;j++){
// 		ran3=(double)rand()/RAND_MAX;
// 		ran4=(double)rand()/RAND_MAX;

// 		double x = xm + asin((2.0 * ran3) -1.0) * (x22-x11)/PI;
// 		double y = ran4 * params.GRID_HEIGHT;

// 		double vx,vy,vz,v= sqrt(2 * fabs(CtoM[0]) * electronTemp);

// 		double z1 = sqrt(-log(ran2()));
// 		double z2 = 6.283185307 * ran2();
// 		vx = v * z1 * cos(z2);
// 		vy = v * z1 * sin(z2);
// 		if(stored == 0){
// 			z1 = sqrt(-log(ran2()));
// 			z2 = 6.283185307 * ran2();
// 			vz = v * z1 * cos(z2);
// 			stored = z1 * sin(z2);
// 		}
// 		else{
// 			vz = v * stored;
// 			stored = 0;
// 		}
// 		h_electrons[i].pointer = -1;
// 		h_ions[i].pointer = -1;
// 		if(x > 0 && x < params.GRID_WIDTH){
// 			h_electrons[i].x  = x;
//             h_electrons[i].y  = y;
// 			h_electrons[i].z = 0;
//             h_electrons[i].vx = vx;
//             h_electrons[i].vy = vy;
//             h_electrons[i].vz = vz;
// 			h_electrons[i].pointer = -1;

// 			v= sqrt(2 * fabs(CtoM[1]) * ionTemp);
// 			double z1 = sqrt(-log(ran2()));
// 			double z2 = 6.283185307 * ran2();
// 			vx = v * z1 * cos(z2);
// 			vy = v * z1 * sin(z2);
// 			if(stored == 0){
// 				z1 = sqrt(-log(ran2()));
// 				z2 = 6.283185307 * ran2();
// 				vz = v * z1 * cos(z2);
// 				stored = z1 * sin(z2);
// 			}
// 			else{
// 				vz = v * stored;
// 				stored = 0;
// 			}
// 			h_ions[i].x  = x;
// 			h_ions[i].y  = y;
// 			h_ions[i].z = 0;
// 			h_ions[i].vx = vx;
// 			h_ions[i].vy = vy;
// 			h_ions[i].vz = vz;
// 			h_ions[i].pointer = -1;
			
// 			i++;
// 		}
// 	}
// 	std::cout << "Generated " << i << " pairs, copying to device." << std::endl;
// 	if((electron_start + (int)fn) > electron.space || (ion_start + (int)fn) > ion.space){
// 		std::cerr << "Error: Exceeding allocated particle space while generating new electron-ion pairs!" << std::endl;
// 		exit(1);
// 	}
// 	q.memcpy(&electron.d_points[electron_start], h_electrons.data(), (int)fn * sizeof(Point)).wait();
// 	q.memcpy(&ion.d_points[ion_start], h_ions.data(), (int)fn * sizeof(Point)).wait();
// 	std::cout << "Copied new pairs to device, updating hashmaps." << std::endl;
// 	q.submit([&](sycl::handler &h) {
// 		h.parallel_for<newPairKernel>(
// 			sycl::range<1>(static_cast<size_t>(i)),
// 			[=](sycl::id<1> idx) {
// 				Point p = electron.d_points[electron_start + idx];
// 				const double inv_dx = 1.0 / params.dx;
// 				const double inv_dy = 1.0 / params.dy;
// 				const int gx = static_cast<int>(p.x * inv_dx);
// 				const int gy = static_cast<int>(p.y * inv_dy);
// 				int grid_key = static_cast<size_t>(gy) * params.GRID_X + gx;
// 				electron.d_points[electron_start + idx].pointer = electron.dmap.insert(grid_key, static_cast<int>(electron_start + idx),electron.NUM_Points);
// 				p = ion.d_points[ion_start + idx];
// 				ion.d_points[ion_start + idx].pointer = ion.dmap.insert(grid_key, static_cast<int>(ion_start + idx),ion.NUM_Points);
// 			}
// 		);
// 	}).wait();
// 	*(electron.NUM_Points) += i;
// 	*(ion.NUM_Points) += i;
// 	ion.END_NUM_Points += i;
// 	electron.END_NUM_Points += i;
// 	std::cout << "Hashmaps updated with new pairs." << std::endl;
// }


void generate_pairs(Particle& electron, Particle& ion, sycl::queue& q, const GridParams& params){
    // Placeholder for pair generation logic
    // This function can be expanded based on specific pairing criteria
    // For now, it does nothing
    //std::cout << "Generating new electron-ion pairs." << std::endl;
    int i=0,j;
    double x,y, ran3, ran4,fn;
    int lower_limit = 0;
    int upper_limit = 1;
    ionimax = 5.23e23;
    fn = params.GRID_HEIGHT * dt * (2.0/PI) * (x22 - x11)  * ionimax *(1.0/(1.67e6));
    if (fn <= 0.0) {
        std::cout << "No pairs to generate (fn <= 0)." << std::endl;
        return;
    }

    // compute integer target (ceiling) and be defensive about attempts
    size_t target_pairs = static_cast<size_t>(std::ceil(fn));
    std::vector<Point> h_electrons;
    std::vector<Point> h_ions;
    h_electrons.reserve(target_pairs);
    h_ions.reserve(target_pairs);

    //std::cout << "Generation started, Number of pairs to generate: " << fn << std::endl;

    // Try up to some multiple of target_pairs attempts to find valid locations
    size_t max_attempts = std::max<size_t>(target_pairs * 10, 100);
    for (size_t attempt = 0; attempt < max_attempts && h_electrons.size() < target_pairs; ++attempt) {
        ran3 = (double)rand() / RAND_MAX;
        ran4 = (double)rand() / RAND_MAX;

        x = xm + asin((2.0 * ran3) - 1.0) * (x22 - x11) / PI;
        y = ran4 * params.GRID_HEIGHT;

        // generate velocities for electron
        double vx_e, vy_e, vz_e;
        double v_e = sqrt(2 * fabs(CtoM[0]) * electronTemp);
        double z1 = sqrt(-log(ran2()));
        double z2 = 6.283185307 * ran2();
        vx_e = v_e * z1 * cos(z2);
        vy_e = v_e * z1 * sin(z2);
        if (stored == 0) {
            z1 = sqrt(-log(ran2()));
            z2 = 6.283185307 * ran2();
            vz_e = v_e * z1 * cos(z2);
            stored = z1 * sin(z2);
        } else {
            vz_e = v_e * stored;
            stored = 0;
        }

        // generate velocities for ion (only if electron location is valid)
        double vx_i, vy_i, vz_i;
        double v_i = sqrt(2 * fabs(CtoM[1]) * ionTemp);
        z1 = sqrt(-log(ran2()));
        z2 = 6.283185307 * ran2();
        vx_i = v_i * z1 * cos(z2);
        vy_i = v_i * z1 * sin(z2);
        if (stored == 0) {
            z1 = sqrt(-log(ran2()));
            z2 = 6.283185307 * ran2();
            vz_i = v_i * z1 * cos(z2);
            stored = z1 * sin(z2);
        } else {
            vz_i = v_i * stored;
            stored = 0;
        }

        if (x > 0.0 && x < params.GRID_WIDTH) {
            Point pe = {};
            pe.x = x; pe.y = y; pe.z = 0;
            pe.vx = vx_e; pe.vy = vy_e; pe.vz = vz_e;
            pe.pointer = -1;
            h_electrons.push_back(pe);

            Point pi = {};
            pi.x = x; pi.y = y; pi.z = 0;
            pi.vx = vx_i; pi.vy = vy_i; pi.vz = vz_i;
            pi.pointer = -1;
            h_ions.push_back(pi);
        }
    }

    i = static_cast<int>(h_electrons.size());
    //std::cout << "Generated " << i << " pairs, copying to device." << std::endl;

    if (i == 0) return;

    if ((electron.END_NUM_Points + i) > electron.space || (ion.END_NUM_Points + i) > ion.space) {
        std::cerr << "Error: Exceeding allocated particle space while generating new electron-ion pairs!" << std::endl;
        exit(1);
    }

    // copy exactly the number actually generated
    q.memcpy(&electron.d_points[electron.END_NUM_Points], h_electrons.data(), static_cast<size_t>(i) * sizeof(Point)).wait();
    q.memcpy(&ion.d_points[ion.END_NUM_Points], h_ions.data(), static_cast<size_t>(i) * sizeof(Point)).wait();

    //std::cout << "Copied new pairs to device, updating hashmaps." << std::endl;
    int iCount = i; // number of pairs actually created
	int e_start = electron.END_NUM_Points; // host value
	int ion_start = ion.END_NUM_Points;

	// Extract device pointers into locals for safe capture
	Point* e_dpoints = electron.d_points;
	Point* ion_dpoints = ion.d_points;
	sycl_hashmap::DeviceView e_dmap = electron.dmap;
	sycl_hashmap::DeviceView ion_dmap = ion.dmap;
	int* e_NUM = electron.NUM_Points; // device pointer
	int* ion_NUM = ion.NUM_Points;
	int e_space = electron.space; // host int
	int ion_space = ion.space;
	const int gridX = params.GRID_X;
	const double inv_dx = 1.0 / params.dx;
	const double inv_dy = 1.0 / params.dy;

	int *error_flag = sycl::malloc_shared<int>(1, q);
	*error_flag = 0;
	q.submit([&](sycl::handler &h) {
		h.parallel_for<class pair_insert_safe>( sycl::range<1>((size_t)iCount),
		[=](sycl::id<1> idx) {
			int local = static_cast<int>(idx[0]);

			int e_idx = e_start + local;
			int ion_idx = ion_start + local;

			// bounds check
			if (e_idx >= e_space || ion_idx >= ion_space) return;

			Point p_e = e_dpoints[e_idx];
			Point p_i = ion_dpoints[ion_idx];

			int gx = static_cast<int>(p_e.x * inv_dx);
			int gy = static_cast<int>(p_e.y * inv_dy);
			if (gx < 0 || gx >= gridX || gy < 0) { // additional check on gy depending on gridY
				e_dpoints[e_idx].pointer = -1;
			} else {
				int grid_key = gy * gridX + gx;
				int ptr_e = e_dmap.insert(grid_key, e_idx, e_NUM);
				e_dpoints[e_idx].pointer = (ptr_e < 0) ? -1 : ptr_e;
				if(ptr_e < 0) {
					*error_flag = 1; // set error flag
				}
			}

			// ion - same grid_key / insertion
			if (gx < 0 || gx >= gridX || gy < 0) {
				ion_dpoints[ion_idx].pointer = -1;
			} else {
				int grid_key = gy * gridX + gx;
				int ptr_i = ion_dmap.insert(grid_key, ion_idx, ion_NUM);
				ion_dpoints[ion_idx].pointer = (ptr_i < 0) ? -1 : ptr_i;
				if(ptr_i < 0) {
					*error_flag = 1; // set error flag
				}
			}
		});
	}).wait();
	if(*error_flag != 0) {
		std::cerr << "Error: Hashmap insertion failed during electron-ion pair generation." << std::endl;
		sycl::free(error_flag, q);
		std::exit(1);
	}
	// Host-side update after successful kernel
	electron.END_NUM_Points += iCount;
	ion.END_NUM_Points += iCount;

    //std::cout << "Hashmaps updated with new pairs." << std::endl;
}
void printTotalEnergy(char *dir, int iteration,double Global_Total_Energy_kin[2],double Global_Total_Energy_pot[2]){


	char fileName[296] = "";
	char fileName1[296] = "";
	char fileName2[296] = "";
	char fileName3[296] = "";

	strcat(fileName, dir);
	strcat(fileName, "/TotalElecKinEnergy.out");
	strcat(fileName1, dir);
	strcat(fileName1, "/TotalIonKinEnergy.out");
	strcat(fileName2, dir);
	strcat(fileName2, "/TotalElecPotEnergy.out");
	strcat(fileName3, dir);
	strcat(fileName3, "/TotalIonPotEnergy.out");

	FILE *fz = fopen(fileName,"a");
	FILE *fa = fopen(fileName1,"a");
	FILE *fb = fopen(fileName2,"a");
	FILE *fc = fopen(fileName3,"a");

	if(fz == NULL) 
	{
		printf("File not Created: PIC/out/IEDFdomain.out\n");
		exit(1);		//PICDAT file not found
	}
	if(fa == NULL) 
	{
		printf("File not Created: PIC/out/source1IEDF.out\n");
		exit(1);		//PICDAT file not found
	}
	
	fprintf(fz,"%d \t",iteration);
	fprintf(fa,"%d \t",iteration);
	fprintf(fb,"%d \t",iteration);
	fprintf(fc,"%d \t",iteration);
	fprintf(fz,"%.16e \n",Global_Total_Energy_kin[0]);
	fprintf(fa,"%.16e \n",Global_Total_Energy_kin[1]);
	fprintf(fb,"%.16e \n",Global_Total_Energy_pot[0]);
	fprintf(fc,"%.16e \n",Global_Total_Energy_pot[1]);


	fclose(fz);
	fclose(fa);
	fclose(fb);
	fclose(fc);


}

void printPhi(char *dir, int iteration){
	int i,j;
	char fileName[296] = "", tmp[100] = "";
	strcat(fileName, dir);
	strcat(fileName, "/potential/");
	sprintf(tmp, "%d", iteration+Iter);
	strcat(fileName, tmp);
	FILE *fd=fopen(fileName,"w");
	if(fd == NULL){
		printf("File not Created: PIC/out/<DIR_NAME>/potential.out\n");
		exit(1);
	}

	for(i=0;i<GRID_Y;i++){
		for(j=0;j<GRID_X-1;j++){
			fprintf(fd,"%lf ",phi[i*GRID_X+j]);
		}
		fprintf(fd,"%lf",phi[i*GRID_X+j]);
		fprintf(fd,"\n");
	}
	fclose(fd);
}

void printElectricField(char *dir, int iteration){
	int i,j;
	/*-----print Ex -------------------------------*/
	char fileName[296] = "", tmp[100] = "";
	strcat(fileName, dir);
	strcat(fileName, "/Ex/");
	sprintf(tmp, "%d", iteration+Iter);
	strcat(fileName, tmp);
	FILE *fd=fopen(fileName,"w");
	if(fd == NULL){
		printf("File not Created: PIC/out/<DIR_NAME>/Ex.out\n");
		exit(1);
	}

	for(i=0;i<GRID_Y;i++){
		for(j=0;j<GRID_X-1;j++){
			fprintf(fd,"%lf ",electricField[i*GRID_X+j].fx);
		}
		fprintf(fd,"%lf",electricField[i*GRID_X+j].fx);
		fprintf(fd,"\n");
	}
	fclose(fd);

	/*-----print Ey -------------------------------*/
	char fileName1[296] = "", tmp1[100] = "";
	strcat(fileName1, dir);
	strcat(fileName1, "/Ey/");
	sprintf(tmp1, "%d", iteration+Iter);
	strcat(fileName1, tmp);
	FILE *fe=fopen(fileName1,"w");
	if(fe == NULL){
		printf("File not Created: PIC/out/<DIR_NAME>/Ey.out\n");
		exit(1);
	}

	for(i=0;i<GRID_Y;i++){
		for(j=0;j<GRID_X-1;j++){
			fprintf(fe,"%lf ",electricField[i*GRID_X+j].fy);
		}
		fprintf(fe,"%lf",electricField[i*GRID_X+j].fy);
		fprintf(fe,"\n");
	}
	fclose(fe);
}

void printEnergy(char *dir, int iteration) {

	int i,j;
	char fileName[296] = "",fileName1[296]="", tmp[100] = "";
	//---Print electron energy in eV ------//
	strcat(fileName, dir);
	strcat(fileName, "/electronenergy/");  //electron energy
	sprintf(tmp, "%d", iteration+Iter);
	strcat(fileName, tmp);
	FILE *fd=fopen(fileName,"w");
	if(fd == NULL){
		printf("File not Created: PIC/out/<DIR_NAME>/electronenergy.out\n");
		exit(1);
	}

	for(i=0;i<GRID_Y;i++){
		for(j=0;j<GRID_X - 1;j++){
			fprintf(fd,"%lf ",0.66667*particles[0].energy_mesh[i*GRID_X+j]/(particles[0].final_mesh[i*GRID_X+j]+1.0));
		}
		fprintf(fd,"%lf\n",0.66667*particles[0].energy_mesh[i*GRID_X+j]/(particles[0].energy_mesh[i*GRID_X+j]+1.0));
	}
	fclose(fd);

	//-----print ion energy in eV---------//

	strcat(fileName1, dir);
	strcat(fileName1, "/ionenergy/");  //electron energy
	sprintf(tmp, "%d", iteration+Iter);
	strcat(fileName1, tmp);
	FILE *fe=fopen(fileName1,"w");
	if(fe == NULL){
		printf("File not Created: PIC/out/<DIR_NAME>/ionenergy.out\n");
		exit(1);
	}

	for(i=0;i<GRID_Y;i++){
		for(j=0;j<GRID_X - 1;j++){
			fprintf(fe,"%lf ",0.66667*particles[1].energy_mesh[i*GRID_X+j]/(particles[1].final_mesh[i*GRID_X+j]+1.0));
		}
		fprintf(fe,"%lf\n",0.66667*particles[1].energy_mesh[i*GRID_X+j]/(particles[1].final_mesh[i*GRID_X+j]+1.0));
	}
	fclose(fe);

}

void printNumberDensity(char *dir, int iteration){
	int i,j;
	char fileName[296] = "";
	char fileName2[296] = "", tmp[100]="", tmp2[100]="";
	strcat(fileName, dir);	
	strcat(fileName, "/electronDensity/");
	sprintf(tmp, "%d", iteration+Iter);
	strcat(fileName, tmp);
	FILE *fd=fopen(fileName,"w");
	if(fd == NULL){
		printf("File not Created: PIC/out/<DIR_NAME>/electronDensity.out\n");
		exit(1);
	}
/*
	for(i=0;i<GRID_Y;i++){
		for(j=0;j<GRID_X;j++){
			fprintf(fd,"%d %d %lf\n",i,j,phi[i*GRID_X+j]);
		}
	}
*/
	for(i=0;i<GRID_Y;i++){
		for(j=0;j<GRID_X-1;j++){
			fprintf(fd,"%lf ",particles[0].final_mesh[i*GRID_X+j]*scaleFactor);
		}
		fprintf(fd,"%lf",particles[0].final_mesh[i*GRID_X+j]*scaleFactor);
		fprintf(fd,"\n");
	}

	fclose(fd);

	strcat(fileName2, dir);
	strcat(fileName2, "/ionDensity/");
	sprintf(tmp2, "%d", iteration+Iter);
	strcat(fileName2, tmp2);
	fd=fopen(fileName2,"w");
	if(fd == NULL){
		printf("File not Created: PIC/out/<DIR_NAME>/ionDensity.out\n");
		exit(1);
	}

	for(i=0;i<GRID_Y;i++){
		for(j=0;j<GRID_X-1;j++){
			fprintf(fd,"%lf ",particles[1].final_mesh[i*GRID_X+j]*scaleFactor);
		}
		fprintf(fd,"%lf",particles[1].final_mesh[i*GRID_X+j]*scaleFactor);
		fprintf(fd,"\n");
	}

	fclose(fd);
}

void printNumParticles(char *dir, int iteration,int* num_particle,int ptype){
	int i,j;
	char fileName[296] = "";
	char fileName2[296] = "", tmp[100]="", tmp2[100]="";
	strcat(fileName, dir);
	if(ptype == 0)
		strcat(fileName, "/electronNumber/");
	else
		strcat(fileName,"/ionNumber/");
	sprintf(tmp, "%d", iteration+Iter);
	strcat(fileName, tmp);
	FILE *fd=fopen(fileName,"w");
	if(fd == NULL){
		printf("File not Created: PIC/out/<DIR_NAME>/electronDensity.out\n");
		exit(1);
	}
/*
	for(i=0;i<GRID_Y;i++){
		for(j=0;j<GRID_X;j++){
			fprintf(fd,"%d %d %lf\n",i,j,phi[i*GRID_X+j]);
		}
	}
*/
	for(i=0;i<GRID_Y;i++){
		for(j=0;j<GRID_X-1;j++){
			fprintf(fd,"%d ",num_particle[i*GRID_X+j]);
		}
		fprintf(fd,"%d",num_particle[i*GRID_X+j]);
		fprintf(fd,"\n");
	}

	fclose(fd);
}