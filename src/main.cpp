#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cstring>

#include "types.hpp"
#include "particles.hpp"
#include "mover.hpp"
#include "interpolation.hpp"
#include "utils.hpp"
#include "sycl_hashmap.hpp"
#include "poissonSolver.hpp"


int main(int argc, char** argv){
    // Argument Checking --> 2 Input Files, Hardware to be used, Work-Group Size, Probing-Size
    if(argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <input_file_p1> <input_file_p2> <cpu|gpu> <wg_size> <prob_size>\n";
        return 1;
    }
    // Argument Parsing
    size_t wg_size = static_cast<size_t>(atoi(argv[4]));
    prob_size = atoi(argv[5]);
    std::cout << "Work-Group Size: " << wg_size << std::endl;
    std::cout << "Probing Size: " << prob_size << std::endl;
    // SYCL Queue Definition and Linking
    sycl::queue q;
    std::string dev_type = argv[3];
    if(dev_type == "cpu") {
        q = sycl::queue(sycl::cpu_selector_v);
    } else if(dev_type == "gpu") {
        q = sycl::queue(sycl::gpu_selector_v);
    } else {
        std::cerr << "Invalid device type. Use 'cpu' or 'gpu'.\n";
        return 1;
    }

    /*Legacy Code*/
    // // Different Types of Species
    // std::vector<Particle> arr;
    // std::string electron = argv[1];
    // Particle pt1;
    // std::strcpy(pt1.name, electron.c_str());
    // std::strcpy(pt1.input_file, electron.c_str());
    // pt1.charge_sign = -1;
    // arr.push_back(pt1);

    // std::string ion = argv[2];
    // Particle pt2;
    // std::strcpy(pt2.name, ion.c_str());
    // std::strcpy(pt2.input_file, ion.c_str());
    // pt2.charge_sign = 1;
    // arr.push_back(pt2);

    // /* 
    // Registering the Species -->
    //     1. Storing the Points of Species in the Array
    //     2. Creation of Map
    //     3. Insertion of Pointers to Original Array in the Map
    // */
    // auto start = std::chrono::high_resolution_clock::now();
    // int Tempcount = 0;
    // readPICDAT("M_PICDAT.DAT",q);
    // CtoM = sycl::malloc_shared<double>(num_type, q);
    // std::cout << "Registering Particles and Inserting into Map" << std::endl;
    // for(auto& pt: arr)
    // {
    //     registerParticle_legacy(pt, q, prob_size, static_cast<int>(wg_size),Tempcount);
    //     Tempcount++;
    // }
    // std::cout << "Particles Registered and Inserted into Map Successfully" << std::endl;
    // // USM Array for Particle(Species) Object -- Data Structure
    // // Particle* particles = sycl::malloc_shared<Particle>(num_type, q);

    // // // // Copying vector of Particles to USM Array of Particles
    // // q.memcpy(particles, arr.data(), num_type*sizeof(Particle));
    // working_size = static_cast<int>(wg_size);
    // GridParams params = arr[0].params;
    // int gridSize = params.GRID_X * params.GRID_Y;
    // std::cout << "Grid Parameters File: " << params.NX << " " << params.NY << " " << params.GRID_X << " " << params.GRID_Y << " " << params.dx << " " << params.dy << std::endl;
    // std::cout << "Grid Parameters DAT: " << params1.NX << " " << params1.NY << " " << params1.GRID_X << " " << params1.GRID_Y << " " << params1.dx << " " << params1.dy << std::endl;
    // std::cout << "File Particles: " << arr[0].END_NUM_Points << " " << arr[1].END_NUM_Points << std::endl;
    // std::cout << "DAT Particles: " << simulationParticles[0] << " " << simulationParticles[1] << std::endl;
    // initVariables_legacy(params, arr, q);
    // // CtoM = sycl::malloc_shared<double>(num_type, q);
    // // corner_mesh = sycl::malloc_device<double>(4 * params.GRID_X * params.GRID_Y, q);
    // // // Creation of Temporary Arrays/Structures on the Device
    // // accessor_s1 = sycl::malloc_device<double>(static_cast<int>(prob_size*gridSize), q);
    // // accessor_s2 = sycl::malloc_device<double>(static_cast<int>(prob_size*gridSize), q);
    // // accessor_s3 = sycl::malloc_device<double>(static_cast<int>(prob_size*gridSize), q);
    // // accessor_s4 = sycl::malloc_device<double>(static_cast<int>(prob_size*gridSize), q);
    // // end_idx = sycl::malloc_device<int>(static_cast<int>(gridSize), q);

    
    /*Legacy Code End Here*/

    /*New Code Start Here*/
    std::vector<std::string> input_files;
    std::string electron = argv[1];
    input_files.push_back(electron);

    std::string ion = argv[2];
    input_files.push_back(ion);
    working_size = static_cast<int>(wg_size);
    auto start = std::chrono::high_resolution_clock::now();
    initVariables("M_PICDAT.DAT", q,input_files);
    /*New Code End Here*/
    auto end = std::chrono::high_resolution_clock::now();

    // Time for Particle Registration and Map Insertion
    double total_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Time Taken for Insertion: " << tt << " s\n";
    std::cout << "Time Taken for Particle Registration and Insertion: " << total_time << " s\n";

    std:: cout << "Charge to Mass Ratio : " << CtoM[0] << " " << CtoM[1] << std::endl;
    // Multiple Charge Deposition and Mover Iterations
    int particle_test[10] = {1,100,4503,5000,480, 9999,60, 6546, 8001, 1080};
    double t1=0, t2=0, t3=0,t4=0,t5 =0,t6=0,t_alloc=0,t_scan=0,t7 = 0;
    FILE* f_particle =fopen("particle_trajectory.out","a");
    
    //int sum_ppc = 0;
    //int* k_idx = sycl::malloc_shared<int>(params1.GRID_X*params1.GRID_Y, q);

    //init Poisson Solver
    initPoissonSolver(q);
    for(int iter = 0; iter < TIMESTEPS; ++iter) {
        std::cout << "Iteration: " << iter << "Particles(e/i): " << particles[0].END_NUM_Points <<" "<< particles[1].END_NUM_Points<< std::endl;
        
        memset(rho,0,params1.GRID_X*params1.GRID_Y*sizeof(double));
        memset(TPENsum,0,num_type*sizeof(double));
        //std::cout << "Starting Charge Deposition" << std::endl;
        for(int i = 0; i < num_type; i++){
            // Charge Deposition
            auto start_charge = std::chrono::high_resolution_clock::now();
            int current_prob_size = calculate_prob_size(q,particles[i].dmap,prob_size,prob_size*working_size,params1);
            interpolate(particles[i].final_mesh, particles[i].energy_mesh,rho, particles[i].d_points, particles[i].params, q, corner_mesh, corner_mesh_enrgy, wg_size, particles[i].dmap, prob_size*working_size, current_prob_size, accessor_s1, accessor_s2, accessor_s3, accessor_s4, accessor_s1_enrgy,accessor_s2_enrgy,accessor_s3_enrgy,accessor_s4_enrgy,particles[i].END_NUM_Points,i);
            q.wait_and_throw();
            auto end_charge = std::chrono::high_resolution_clock::now();
            t1 += std::chrono::duration<double>(end_charge - start_charge).count();
        }
        //std::cout << "Charge Deposition Complete, Poisson Solver started" << std::endl;
        //Poisson Solver
        auto start_poisson = std::chrono::high_resolution_clock::now();
        poissonSolver(rho, phi, iter, q);
        auto end_poisson = std::chrono::high_resolution_clock::now();
        t4 += std::chrono::duration<double>(end_poisson - start_poisson).count();
        //std::cout << "Poisson Solver Complete, Starting Mover" << std::endl;
        //q.memcpy(k_idx, particles[i].dmap.key_idx, sizeof(int)*params1.GRID_X*params1.GRID_Y).wait();
        for(int i = 0; i < num_type; i++){
            // q.memcpy(k_idx, particles[i].dmap.key_idx, sizeof(int)*params1.GRID_X*params1.GRID_Y).wait();
            // for(int j = 0; j < params1.GRID_X*params1.GRID_Y; j++)
            // {
            //     sum_ppc += k_idx[j];
            // }
            //std::cout << "Particle per cell avg for species " << i << " : " << sum_ppc/(double)(params1.GRID_X*params1.GRID_Y) << std::endl;
            //sum_ppc = 0;
            //Charge Mover
            auto start_mover = std::chrono::high_resolution_clock::now();
            mover(particles[i].d_points, params1, q, particles[i].dmap, prob_size*working_size, iter, end_idx,prob_size, particles[i].NUM_Points,particles[i].empty_space,particles[i].empty_space_idx,i,particles[i].END_NUM_Points,particles[i].new_additions);
            q.wait_and_throw();
            auto end_mover = std::chrono::high_resolution_clock::now();
            t2 += std::chrono::duration<double>(end_mover - start_mover).count();
            // // Map Capacity Check -- Debugging Purpose
            // q.memcpy(k_idx, particles[i].dmap.key_idx, sizeof(int)*params1.GRID_X*params1.GRID_Y).wait();
            // for(int j = 0; j < params1.GRID_X*params1.GRID_Y; j++)
            // {
            //     sum_ppc += k_idx[j];
            // }
            // std::cout << "Particle per cell avg for species " << i << " : " << sum_ppc/(double)(params1.GRID_X*params1.GRID_Y) << std::endl;
            // sum_ppc = 0;
            // for(int j = 0;j < 20;j++)
            //     std::cout << k_idx[j] << std::endl;
            // for(int j = 0; j < params1.GRID_X*params1.GRID_Y; j++)
            // {
            //     if(k_idx[j] >= prob_size*working_size)
            //     {
            //         std::cerr << "Error: Hashmap capacity exceeded for species " << i << " at grid " << j << " with key_idx " << k_idx[j] << std::endl;
            //         exit(1);
            //     }
            // }
            // std::cout <<sum_ppc <<", " << sum_ppc/(double)(params1.GRID_X*params1.GRID_Y) << " particles per cell on avg for species " << i << std::endl;
            // sum_ppc = 0;
            //Printing Total Energy
            auto print_time = std::chrono::high_resolution_clock::now();
            // if(iter % printInterval == 0)
            // {
            //     Total_energy_kinetic[i] = calculate_energy(q,particles[i].d_points,particles[i].END_NUM_Points,CtoM,i);
            //     //Total_energy_kinetic[i] = calc_energy_serial(particles[i].d_points,particles[i].END_NUM_Points,CtoM,i);
            // }
            auto end_print_time = std::chrono::high_resolution_clock::now();
            t6 += std::chrono::duration<double>(end_print_time - print_time).count();
            // for(int k = 0; k < 10; k++)
            // {
            //     int idx = particle_test[k];
            //     if(idx < particles[i].END_NUM_Points)
            //     {
            //         Point p;
            //         q.memcpy(&p, &particles[i].d_points[idx], sizeof(Point)).wait();
            //         fprintf(f_particle,"%d \t %d \t %d \t %d \t %lf \t %lf\t %lf\t %lf\n",iter,i,idx,p.pointer,p.x,p.y,p.vx,p.vy);
            //     }
            // }

            //Particle Compaction
            auto start_compact = std::chrono::high_resolution_clock::now();
            if(iter % printInterval == 0)
            {
                particles[i].END_NUM_Points = compact_particles(particles[i].d_points, particles[i].dmap, q, particles[i].empty_space, particles[i].empty_space_idx,particles[i].END_NUM_Points,particles[i].NUM_Points, t_alloc,t_scan);
                q.wait_and_throw();
            }
            
            auto end_compact = std::chrono::high_resolution_clock::now();
            t5 += std::chrono::duration<double>(end_compact - start_compact).count();
            // Map Cleanup and Compaction
            /** DONT CHANGE - CAUTION **/
            /** Do not disturb this code until you figure out prob_size* work_size is large enough **/
            /****/
            auto start_clean = std::chrono::high_resolution_clock::now();
            if(iter % 1 == 0)
            {
                cleanup_map(params1, q, particles[i].dmap,particles[i].d_points);
                q.wait_and_throw();
            }
            auto end_clean = std::chrono::high_resolution_clock::now();
            t3 += std::chrono::duration<double>(end_clean - start_clean).count();
            
        }
        //std::cout << "Main Operations Complete for Iteration, Generating New Ions/Electrons" << std::endl;
        auto start_newpart = std::chrono::high_resolution_clock::now();
        int num_ele = 0;
        int num_ion = 0;
        q.memcpy(&num_ele, particles[0].new_additions,sizeof(int)).wait();
        q.memcpy(&num_ion, particles[1].new_additions,sizeof(int)).wait();
        int num_new = num_ele - num_ion;
        if(num_new < 0) num_new = 0;
        q.memset(particles[0].new_additions,0,sizeof(int)).wait();
        q.memset(particles[1].new_additions,0,sizeof(int)).wait();
        //std::cout << "Number of New Electrons to be Generated: " << num_new << std::endl;
        particles[0].END_NUM_Points = generate_new_electrons(particles[0].d_points, q, params1, particles[0].dmap, particles[0].END_NUM_Points, particles[0].NUM_Points, particles[0].space, CtoM, electronTemp, stored,num_new);
        q.wait_and_throw();
        generate_pairs(particles[0], particles[1], q, params1);
        q.wait_and_throw();
        auto end_newpart = std::chrono::high_resolution_clock::now();
        t7 += std::chrono::duration<double>(end_newpart - start_newpart).count();
        // // Printing Work
        if(iter % printInterval == 0)
        {
            //printTotalEnergy(".",iter,Total_energy_kinetic,Total_energy_Pot);
            printNumberDensity("./out",iter);
            printEnergy("./out",iter);
            printElectricField("./out",iter);
            printPhi("./out",iter);
        }
    }
    fclose(f_particle);
    // Saving the final Grid Output --> Charge Deposition Scheme
    for(int i = 0; i < num_type; i++){
        std::string file = "Mesh_" + std::to_string(i) + ".out";
        std::ofstream outfile(file);
        if (!outfile.is_open()) {
            std::cerr << "Error opening output file: " << file << std::endl;
        } else {
            for(int y = 0; y < particles[i].params.GRID_Y; ++y) {
                for(int x = 0; x < particles[i].params.GRID_X; ++x) {
                    outfile << particles[i].final_mesh[static_cast<size_t>(y) * params1.GRID_X + x] << " ";
                }
                outfile << "\n";
            }
            outfile.close();
        }
    }

    // Freeing the Temporary Data
    // sycl::free(corner_mesh, q);
    // sycl::free(accessor_s1, q);
    // sycl::free(accessor_s2, q);
    // sycl::free(accessor_s3, q);
    // sycl::free(accessor_s4, q);
    // sycl::free(end_idx, q);
    freeVariables(q);
    // Time Taken by Different Processes
    std::cout << "Charge Deposition time: " << t1 << " seconds\n";
    std::cout << "Mover time: " << t2 << " seconds\n";
    std::cout << "Particle Compaction time: " << t5 << " seconds\n";
    std::cout << "Allocation time in Compaction: " << t_alloc << " seconds\n";
    std::cout << "Scan time in Compaction: " << t_scan << " seconds\n";
    std::cout << "Map Cleanup time: " << t3 << " seconds\n";
    std::cout << "Poisson Solver time: " << t4 << " seconds\n";
    std::cout << "Print time: " << t6 << " seconds\n";
    std::cout << "New Particle Generation time: " << t7 << " seconds\n";
    std::cout << "Total execution time: " << t1+t2+t3+t4+t5+t6+t7 << " seconds\n";

    return 0;
}
