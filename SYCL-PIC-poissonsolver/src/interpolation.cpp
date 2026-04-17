#include "interpolation.hpp"
#include <algorithm>
#include <cmath>
#include <sycl/sycl.hpp>
#include "sycl_hashmap.hpp"


int calculate_working_size(int prob_size,int probing)
{
    int working_size = std::ceil((double)probing/prob_size);
    return working_size;
}
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
                 int ptype)
{
    const size_t gridSize = static_cast<size_t>(params.GRID_X) * params.GRID_Y;
    //const size_t numPoints = params.NUM_Points;

    q.parallel_for<PartialMeshInitKernel>(
        sycl::range<1>(4 * gridSize),
        [=](sycl::id<1> idx) {
            corner_mesh[idx[0]] = 0.0;
            corner_mesh_enrgy[idx[0]] = 0.0;
        }
    ).wait();

    size_t global_size = gridSize*prob_size;
    //int working_size = probing_length/prob_size;
    const double denom = 1/fabs(CtoM[ptype]);
    Total_energy[ptype] = 0;
    double TPEN = 0;
    int TPDN = 0;
    const double weighing = weighingFactor[ptype];
    std::cout << "Current Prob Size: " << prob_size << " for type:" << ptype << std::endl;
    //std::cout << "Starting Inerpolation\n";
    q.submit([&](sycl::handler& h) {
        h.parallel_for<InterpolationKernel>(
            sycl::nd_range<1>(global_size, prob_size),
            [=](sycl::nd_item<1> item) {
                int global_id = item.get_global_id(0);
                int grid_global_idx = global_id / prob_size;
                int local_id = item.get_local_id(0);
                if(grid_global_idx > gridSize) return;

                int k_idx = dmap.key_idx[grid_global_idx];
                const int working_size = calculate_working_size(prob_size,k_idx);
                int idx = local_id * working_size;
    

                int *points = dmap.retrieve(static_cast<int>(grid_global_idx));
                if(points == NULL) return;
                const int gx = grid_global_idx % params.GRID_X;
                const int gy = grid_global_idx / params.GRID_X;
                int limit = std::min(k_idx, (local_id + 1) * working_size);

                double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
                double s1_e = 0, s2_e = 0, s3_e = 0, s4_e = 0;
                for (int i = idx; i < limit; i++) {
                    if(dmap.keys[i + grid_global_idx*probing_length] == 0 || points[i] == -1) continue;
                    Point p = points_device[points[i]];
                    double px = p.x;
                    double py = p.y;
                    double pvx = p.vx;				//particle x velocity
			        double pvy = p.vy;				//particle y velocity
			        double pvz = p.vz;
                    double lx = px - gx * params.dx;
                    double ly = py - gy * params.dy;
                    double v2 = pvx*pvx + pvy*pvy + pvz*pvz;
                    double enrgy = 0.5 * v2 * denom;
                    s1 += (params.dx - lx) * (params.dy - ly);
                    s2 += lx * (params.dy - ly);
                    s3 += (params.dx - lx) * ly;
                    s4 += lx * ly;

                    s1_e += ((params.dx - lx) * (params.dy - ly))*enrgy;
                    s2_e += (lx * (params.dy - ly))*enrgy;
                    s3_e += (params.dx - lx) * ly * enrgy;
                    s4_e += lx * ly * enrgy;
                }
                accessor_s1[global_id] = s1;
                accessor_s2[global_id] = s2;
                accessor_s3[global_id] = s3;
                accessor_s4[global_id] = s4;

                accessor_s1_enrgy[global_id] = s1_e;
                accessor_s2_enrgy[global_id] = s2_e;
                accessor_s3_enrgy[global_id] = s3_e;
                accessor_s4_enrgy[global_id] = s4_e;
            }
        );
    }).wait();

    q.parallel_for<InterReductionKernel>(
        sycl::range<1>(gridSize),
        [=](sycl::id<1> item) {
            const size_t cell_global_idx = item[0];
            const int gx = cell_global_idx % params.GRID_X;
            const int gy = cell_global_idx / params.GRID_X;
            if ((gx + 1) >= params.GRID_X || (gy + 1) >= params.GRID_Y) return;
            int p1 = gy * params.GRID_X + gx;
            int p2 = gy * params.GRID_X + (gx + 1);
            int p3 = (gy + 1) * params.GRID_X + gx;
            int p4 = (gy + 1) * params.GRID_X + (gx + 1);
            int a1 = cell_global_idx * prob_size;
            int a2 = a1 + prob_size;
            double r1 = 0, r2 = 0, r3 = 0, r4 = 0;
            double r1_e = 0, r2_e = 0, r3_e = 0, r4_e = 0;
            for(int i = a1; i < a2; i++)
            {
                r1 += accessor_s1[i];
                r2 += accessor_s2[i];
                r3 += accessor_s3[i];
                r4 += accessor_s4[i];
                r1_e += accessor_s1_enrgy[i];
                r2_e += accessor_s2_enrgy[i];
                r3_e += accessor_s3_enrgy[i];
                r4_e += accessor_s4_enrgy[i];
            }
            corner_mesh[0 * gridSize + p1] = r1 * weighing;
            corner_mesh[1 * gridSize + p2] = r2 * weighing;
            corner_mesh[2 * gridSize + p3] = r3 * weighing;
            corner_mesh[3 * gridSize + p4] = r4 * weighing;
            corner_mesh_enrgy[0 * gridSize + p1] = r1_e * weighing;
            corner_mesh_enrgy[1 * gridSize + p2] = r2_e * weighing;
            corner_mesh_enrgy[2 * gridSize + p3] = r3_e * weighing;
            corner_mesh_enrgy[3 * gridSize + p4] = r4_e * weighing;
        }
    ).wait();

    q.parallel_for<ReducePartialMeshesKernel>(
        sycl::range<1>(gridSize),
        [=](sycl::id<1> cell_global_id_obj) {
            const size_t cell_global_idx = cell_global_id_obj[0];
            double sum = 0.0;
            double sum_e = 0.0;
            for (int k_buf = 0; k_buf < 4; ++k_buf) {
                sum += corner_mesh[k_buf * gridSize + cell_global_idx];
                sum_e += corner_mesh_enrgy[k_buf * gridSize + cell_global_idx];
            }
            final_mesh[cell_global_idx] = sum;
            energy_mesh[cell_global_idx] = sum_e;
        }
    ).wait();
    double chrg = -particleCharges[ptype] / epsilon;
    q.parallel_for<RhoKernel>(
        sycl::range<1>(gridSize),
        [=](sycl::id<1> item) {
            const size_t idx = item[0];
            rho[idx] += final_mesh[idx] * chrg;
        }
    ).wait();
    // for(int i = 0;i < 100;i++){
    //     std::cout << energy_mesh[i] << " ";
    // }
    //std::cout <<" Complete\n" <<std::endl;
}
