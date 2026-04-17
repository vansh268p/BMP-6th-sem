#include "mover.hpp"
#include <oneapi/mkl/rng/device.hpp>
#include <algorithm>
#include <sycl/sycl.hpp>
#include "sycl_hashmap.hpp"
#include "types.hpp"
#include <math.h>
#include <thread>
#include "utils.hpp"

namespace rngd = oneapi::mkl::rng::device;

double modulo(double x, double y){

	if(x < 0) {
		if((int)(x/y) ==  x/y ) return 0;
		return y-fmod(-x,y);
	}
	else {
		return fmod(x,y);
	}
}
inline int random_mover(Point *p, int idx, int iter)
{
    std::uint64_t seed = 12345ULL * iter + idx * 67890123ULL;

    // Engine
    rngd::philox4x32x10<4> engine(seed, idx);

    // Distributions
    rngd::uniform<double> distr(0.0, 1.0);

    // Generate random number for probability decision
    double rand_prob = rngd::generate_single(distr, engine);

    if (rand_prob < p_val) {
        // Generate two random numbers between 0 and 1
        p->x = rngd::generate_single(distr, engine);
        p->y = rngd::generate_single(distr, engine);
        return 0;
    }

    return 1;
}

inline void deterministic_mover_vec(
        Point        *p,
        int           ptype,
        const double *CtoM,            // q / m lookup table
        double        dt,
        const Field  *electricField,   // still zeros in your code
        const double *magneticField,   //  "
        const GridParams &params)
{
    using vec2 = sycl::vec<double,2>;

    /* === gather particle state === */
    vec2   pos { p->x,  p->y };
    vec2   vel { p->vx, p->vy };
    double pz  = p->z;
    double vz  = p->vz;

    /* === constants === */
    const double pctm = CtoM[ptype];           // (q/m)
    const int    gx   = static_cast<int>(pos.x() / params.dx);
    const int    gy   = static_cast<int>(pos.y() / params.dy);

    /* === interpolate fields (still placeholders) === */
    double Ex = 0.0;
    double Ey = 0.0;
    double Bz = 0.0;

    /* === Boris push (vector form) ================== */
    const vec2  half_qmE  = vec2(Ex,Ey) * (0.5 * pctm * dt);
    vec2 v_minus = vel + half_qmE;

    const double t  = 0.5 * pctm * Bz * dt;
    const double s  = 2.0 * t / (1.0 + t * t);

    /* rotate in-plane velocity */
    vec2 v_prime { v_minus.x() + v_minus.y()*t,
                   v_minus.y() - v_minus.x()*t };

    vec2 v_plus = v_minus + vec2( v_prime.y()*s,
                                 -v_prime.x()*s );

    vec2 vel_new = v_plus + half_qmE;

    /* === update position =========================== */
    pos += vel_new * dt;
    pz  += vz * dt;

    /* === write-back =================================*/
    p->x  = pos.x();
    p->y  = pos.y();
    p->z  = pz;
    p->vx = vel_new.x();
    p->vy = vel_new.y();
    p->vz = vz;          // unchanged in your model
}


inline void deterministic_mover(Point& p, int ptype,const double *CtoM, const double dt, const Field *electricField, const double* magneticField,const GridParams& params, const double* phi){
    double px = p.x;
    double py = p.y;
    //printf("I got y as %lf\n",py);
    double pz = p.z;
    double pvx = p.vx;
    double pvy = p.vy;
    double pvz = p.vz;
    //int ptype = particles[i].type;
    double pctm = CtoM[ptype];
                    
                //printf("ctom:%lf\n",CtoM[1]);
    int gx = (int) (px/params.dx);
    int gy = (int) (py/params.dy);
    double lx=px - gx * params.dx;
    double ly=py - gy * params.dy;


    /* linear interpolation to find x and y coordinates of the fields on the particles*/
    
    double Epx = (electricField[gy*params.GRID_X+gx].fx * (params.dx-lx) * (params.dy-ly) + electricField[gy*params.GRID_X+(gx+1)].fx * lx * (params.dy-ly) + 
                electricField[(gy+1)*params.GRID_X+gx].fx * (params.dx-lx) * ly + electricField[(gy+1)*params.GRID_X+(gx+1)].fx * lx * ly)/(params.dx * params.dy);

    double Epy = (electricField[gy*params.GRID_X+gx].fy * (params.dx-lx) * (params.dy-ly) + electricField[gy*params.GRID_X+(gx+1)].fy * lx * (params.dy-ly) + 
                electricField[(gy+1)*params.GRID_X+gx].fy * (params.dx-lx) * ly + electricField[(gy+1)*params.GRID_X+(gx+1)].fy * lx * ly)/(params.dx * params.dy);

    double Bpz = (magneticField[gy*params.GRID_X+gx] * (params.dx-lx) * (params.dy-ly) + magneticField[gy*params.GRID_X+(gx+1)] * lx * (params.dy-ly) + 
                magneticField[(gy+1)*params.GRID_X+gx] * (params.dx-lx) * ly + magneticField[(gy+1)*params.GRID_X+(gx+1)] * lx * ly)/(params.dx * params.dy);

    double pot = (phi[gy*params.GRID_X+gx] * (params.dx-lx) * (params.dy-ly) + phi[gy*params.GRID_X+(gx+1)] * lx * (params.dy-ly) + 
                phi[(gy+1)*params.GRID_X+gx] * (params.dx-lx) * ly + phi[(gy+1)*params.GRID_X+(gx+1)] * lx * ly)/(params.dx * params.dy);			

    // double Epx = 0;
    // double Epy = 0;
    // double Bpz = 0;
    // double pot = 0;

    
    // #pragma omp atomic
    // Total_energy_Pot[ptype]+=pot*particleCharges[ptype]*1e5;
        //printf("Potential Energy of %d = %.16e\n ",ptype,Total_energy_Pot[ptype]);
    


    /* boris method implementation (de facto standerd for calculating velocity and position of charged particles)
        for boris method refer to following link: https://www.particleincell.com/2011/vxb-rotation/ 
    */

    double Vxminus = pvx + 0.5 * pctm * Epx * dt;
    double Vyminus = pvy + 0.5 * pctm * Epy * dt; 

    double t = 0.5 * pctm * Bpz * dt;

    double Vxprime = Vxminus + Vyminus * t;								
    double Vyprime = Vyminus - Vxminus * t;

    double s = 2 * t / (1 + t * t);

    pvx = Vxminus + Vyprime * s + 0.5 * pctm * Epx * dt;
    pvy = Vyminus - Vxprime * s + 0.5 * pctm * Epy * dt;

    px =px + dt * pvx;
    py =py + dt * pvy;
    pz =pz + dt * pvz;

    p.x =px;
    p.y =py;
    p.z =pz;

    p.vx = pvx ;
    p.vy = pvy ;
    p.vz = pvz ;

}


void mover(Point* points_device,
           const GridParams& params,
           sycl::queue& q,
           sycl_hashmap::DeviceView& dmap,
           const int probing_length,
           int iter,
           int* end_idx,
           int prob_size,
           int* NUM_Points,
           int* empty_space,
           int* empty_space_idx,
           int ptype,
           int num_particles,
           int* new_additions)
{
    int gridSize = params.GRID_X * params.GRID_Y;
    int global_size = gridSize * probing_length;
    const int periodicBoundary_local = periodicBoundary;
    const double *CtoM_local = CtoM;
    const double dt_local = dt;
    const Field* electric_field_local = electricField;
    const double* magnetic_field_local = magneticField;
    const double* phi_local = phi;
    // int* empty_space_local = empty_space;
    // int *empty_space_count = empty_space_idx;
    q.parallel_for<TempKernel>(
        sycl::range<1>(gridSize),
        [=](sycl::id<1> item) {
            const size_t i = item[0];
            end_idx[i] = dmap.key_idx[i];
        }
    ).wait();
    /*Original Code Starts Here*/
    // q.submit([&](sycl::handler& h) {
    //     h.parallel_for<MoverKernel>(
    //         sycl::nd_range<1>(global_size, prob_size),
    //         [=](sycl::nd_item<1> item) {

    //             // Identify the ID of the Grid
    //             int global_id = item.get_global_id(0);
    //             int grid_global_idx = global_id / probing_length;
    //             int local_id = global_id % probing_length; 
    //             if(grid_global_idx > gridSize) return;

    //             int *points = dmap.retrieve(static_cast<int>(grid_global_idx));
    //             if(points == NULL || dmap.keys[global_id] != 1 || local_id >= end_idx[grid_global_idx]) return;
    //             Point *p = &points_device[points[local_id]];
    //             int value = points[local_id];
    //             int gx_before = static_cast<int>(p->x / params.dx);
    //             int gy_before = static_cast<int>(p->y / params.dy);
    //             /*Legacy Code */
    //             // //Mover Logic
    //             // if(!random_mover(p, global_id, iter))
    //             // {
    //             //     dmap.deletion(global_id,NUM_Points);
    //             //     p->pointer = -1;
    //             //     const double inv_dx = 1.0 / params.dx;
    //             //     const double inv_dy = 1.0 / params.dy;
    //             //     const int gx = static_cast<int>(p->x * inv_dx);
    //             //     const int gy = static_cast<int>(p->y * inv_dy);
    //             //     int grid_key = static_cast<size_t>(gy) * params.GRID_X + gx;
    //             //     p->pointer = dmap.insert(grid_key, value,NUM_Points);
    //             // }
    //             /*Legacy Code End Here*/
    //             deterministic_mover(p,ptype,CtoM_local,dt_local,electric_field_local,magnetic_field_local,params);
    //             double px = p->x;
    //             double py = p->y;
    //             int gy_after = static_cast<int>(py / params.dy);
    //             int gx_after = static_cast<int>(px / params.dx);
    //             if(ptype == 0){
	// 				if(periodicBoundary_local){

	// 					py = modulo( py , params.GRID_HEIGHT);
	// 					p->y =py;
	// 				}
	// 				if( px >= params.GRID_WIDTH || px < 0.0 || py>= params.GRID_HEIGHT || py < 0){
    //                     dmap.deletion(global_id,NUM_Points);
    //                     p->pointer = -1;
    //                     // if(doReinjection){
						
	// 					// 	counter_for_add[ id ] +=1;
						
	// 					// }
	// 				}else if(gx_before != gx_after || gy_before != gy_after){
    //                     dmap.deletion(global_id,NUM_Points);
    //                     p->pointer = -1;
    //                     const double inv_dx = 1.0 / params.dx;
    //                     const double inv_dy = 1.0 / params.dy;
    //                     const int gx = static_cast<int>(p->x * inv_dx);
    //                     const int gy = static_cast<int>(p->y * inv_dy);
    //                     int grid_key = static_cast<size_t>(gy) * params.GRID_X + gx;
    //                     p->pointer = dmap.insert(grid_key, value,NUM_Points);
    //                 }
	// 		    }else{
    //                 if(periodicBoundary_local){
    //                     // py = modulo( (py + dt * pvy) , GRID_HEIGHT);
    //                     py = modulo( py , params.GRID_HEIGHT);
    //                     p->y =py;
    //                 }
	// 			    if( px >= params.GRID_WIDTH || px < 0.0 || py>= params.GRID_HEIGHT || py < 0){
    //                     dmap.deletion(global_id,NUM_Points);
    //                     p->pointer = -1;
	// 				}else if(gx_before != gx_after || gy_before != gy_after){
    //                     dmap.deletion(global_id,NUM_Points);
    //                     p->pointer = -1;
    //                     const double inv_dx = 1.0 / params.dx;
    //                     const double inv_dy = 1.0 / params.dy;
    //                     const int gx = static_cast<int>(p->x * inv_dx);
    //                     const int gy = static_cast<int>(p->y * inv_dy);
    //                     int grid_key = static_cast<size_t>(gy) * params.GRID_X + gx;
    //                     p->pointer = dmap.insert(grid_key, value,NUM_Points);
    //                 }
	// 		    }
    //         }
    //     );
    // }).wait();
    /*Original Code Ends Here*/

    /*New Code starts Here*/
    //std:: cout << "Starting Mover Kernel\n";
    int* mover_left_local = particles[ptype].mover_left;
    int* mover_idx_local = particles[ptype].mover_left_idx;
    int* error_flag = malloc_shared<int>(1,q);
    //int *overflow_index = malloc_shared<int>(1,q);
    *error_flag = 0;
    q.parallel_for<MoverKernel>(
        sycl::range<1>(num_particles),              // num_particles = params.NUM_Points
        [=](sycl::id<1> idx) {
            Point p = points_device[idx];
            if(p.pointer == -1) return;
            const double inv_dx = 1.0 / params.dx;
            const double inv_dy = 1.0 / params.dy;
            int gx_before = static_cast<int>(p.x * inv_dx);
            int gy_before = static_cast<int>(p.y * inv_dy);
            deterministic_mover(p,ptype,CtoM_local,dt_local,electric_field_local,magnetic_field_local,params,phi_local);
            //deterministic_mover_vec(p,ptype,CtoM_local,dt_local,electric_field_local,magnetic_field_local,params);
            double px = p.x;
            double py = p.y;
            if(periodicBoundary_local){
                py = modulo( py , params.GRID_HEIGHT);
                p.y =py;
            }
            int global_id = p.pointer;
            int value = idx;
            int gy_after = static_cast<int>(py * inv_dy);
            int gx_after = static_cast<int>(px * inv_dx);
            bool out_of_bounds = (px >= params.GRID_WIDTH || px < 0.0 || py >= params.GRID_HEIGHT || py < 0);
            bool change_cell = (gx_before != gx_after || gy_before != gy_after);
            // if(periodicBoundary_local){
            //     py = modulo( py , params.GRID_HEIGHT);
            //     p.y =py;
            // }
            if(out_of_bounds){
                dmap.deletion(global_id,NUM_Points);
                p.pointer = -1;
                if(!store_deleted_indices(idx,empty_space,empty_space_idx, num_particles)){
                }

                if(ptype == 0){
                    // if(doReinjection){
                        
                    //     counter_for_add[ id ] +=1;
                        
                    // }
                }
                if(px < 0.0) {
                    // Particle went out on the left side
                    auto atomic_count = sycl::atomic_ref<int,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>( *new_additions );
                    atomic_count.fetch_add(1);
                    // Here, you can store additional information if needed for reinjection
                }
            }else if(change_cell){
                dmap.deletion(global_id,NUM_Points);
                p.pointer = -1;
                int grid_key = static_cast<size_t>(gy_after) * params.GRID_X + gx_after;
                p.pointer = dmap.insert(grid_key, value,NUM_Points);
                if(p.pointer == -1)
                {
                    *error_flag = 1;
                    store_deleted_indices(idx,mover_left_local,mover_idx_local,num_particles);
                    //*overflow_index = grid_key;
                }
            }
            points_device[idx] = p;
        }
    ).wait();
    int* k_idx = sycl::malloc_shared<int>(params.GRID_X*params.GRID_Y,q);
    if(*error_flag != 0)
    {
        int num_insertions = *mover_idx_local;
        *error_flag = 0;
        q.submit([&](sycl::handler& h) {
            h.parallel_for<class MoverInsertions>( sycl::range<1>((size_t)num_insertions),
            [=](sycl::id<1> idx) {
                if(*error_flag != 0) return; // early out on error
                int local = static_cast<int>(idx[0]);
                int global_idx = mover_left_local[local]; // target index in point_device
                
                // bounds check to be safe (shouldn't happen due to earlier check)
                if (global_idx < 0 || global_idx >= num_particles) return;

                Point p = points_device[global_idx]; // local copy
                const double inv_dx = 1.0 / params.dx;
                const double inv_dy = 1.0 / params.dy;

                int gx = static_cast<int>(p.x * inv_dx);
                int gy = static_cast<int>(p.y * inv_dy);

                // validate grid indices
                // if (gx < 0 || gx >= params.GRID || gy < 0 || gy >= gridY) {
                //     // invalid grid -> do not insert; mark pointer -1 so compaction can pick it up
                //     points_device[global_idx].pointer = -1;
                //     return;
                // }

                int grid_key = gy * params.GRID_X + gx;

                // Insert into hashmap. dmap.insert should be safe and expects device-accessible NUM_Points.
                int new_ptr = dmap.insert(grid_key, global_idx, NUM_Points);
                // dmap.insert may return -1 on failure. Handle it.
                if (new_ptr < 0) {
                    // insertion failed; mark pointer invalid and return
                    points_device[global_idx].pointer = -1;
                    *error_flag = 1; // set error flag
                } else {
                    // store pointer returned by hashmap
                    points_device[global_idx].pointer = new_ptr;
                }
            });
        }).wait(); // ensure kernel completes before host reads NUM_Points or runs compaction
        if(*error_flag != 0) {
            q.memcpy(k_idx,dmap.key_idx,params.GRID_X*params.GRID_Y*sizeof(int));
            printNumParticles("./out",iter,k_idx,ptype);
            std::cerr << "Error: Hashmap insertion failed during Insertion, Map Overflow." << std::endl;
            sycl::free(error_flag, q);
            sycl::free(k_idx,q);
            std::exit(1);
        }
    }
    if(iter % printInterval == 0)
    {
        q.memcpy(k_idx,dmap.key_idx,params.GRID_X*params.GRID_Y*sizeof(int));
        printNumParticles("./out",iter,k_idx,ptype);
    }
    sycl::free(k_idx,q);
    //std:: cout << "Mover Kernel Completed\n";
    //size_t num_workers = std::max<size_t>(1u, std::thread::hardware_concurrency());
//     size_t num_workers = 1024;
// // choose chunked global size = num_workers (each item does a contiguous chunk)
//     size_t global_workers = num_workers;
//     size_t local_size = 64;

//     q.parallel_for<MoverKernel>(
//         sycl::nd_range<1>(sycl::range<1>(global_workers), sycl::range<1>(local_size)),
//         [=](sycl::nd_item<1> item) {
//         size_t worker = item.get_global_id(0);
//         size_t N = (size_t) num_particles;
//         size_t chunk = (N + global_workers - 1) / global_workers;
//         size_t start = worker * chunk;
//         size_t end = std::min(start + chunk, N);

//         // precompute some constants once per worker
//         const double inv_dx = 1.0 / params.dx;
//         const double inv_dy = 1.0 / params.dy;

//         for (size_t idx = start; idx < end; ++idx) {
//             Point *p = &points_device[idx];
//             if (p->pointer == -1) continue;

//             int gx_before = static_cast<int>(p->x * inv_dx);
//             int gy_before = static_cast<int>(p->y * inv_dy);
//             deterministic_mover(p, ptype, CtoM_local, dt_local, electric_field_local, magnetic_field_local, params);

//             double px = p->x;
//             double py = p->y;
//             int global_id = p->pointer;
//             int value = (int)idx;
//             int gy_after = static_cast<int>(py * inv_dy);
//             int gx_after = static_cast<int>(px * inv_dx);
//             bool out_of_bounds = (px >= params.GRID_WIDTH || px < 0.0 || py >= params.GRID_HEIGHT || py < 0);
//             bool change_cell = (gx_before != gx_after || gy_before != gy_after);

//             if (periodicBoundary_local) {
//                 py = modulo(py, params.GRID_HEIGHT);
//                 // write only if changed (avoid useless store)
//                 if (py != p->y) p->y = py;
//             }

//             if (out_of_bounds) {
//                 dmap.deletion(global_id, NUM_Points);
//                 p->pointer = -1;
//             } else if (change_cell) {
//                 dmap.deletion(global_id, NUM_Points);
//                 p->pointer = -1;
//                 int grid_key = gy_after * params.GRID_X + gx_after;
//                 p->pointer = dmap.insert(grid_key, value, NUM_Points);
//             }
//         }
//     }).wait();

    /*New Code ends Here*/
}