#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
#include <random>
#include <vector>

struct Element {
    double x;
    double y;
    int pointer;
};

class ScanAndRecordKernel;
class SwapKernel;

void compactArrayDualPointer(sycl::queue& q, Element* array, int* gap_positions, 
                              int* valid_positions, int n, int num_valid, 
                              double& kernel_time_ms) {
    
    // Allocate two atomic counters (local to this function)
    int* d_left_write = sycl::malloc_device<int>(1, q);   // Counter for gaps found
    int* d_right_scan = sycl::malloc_device<int>(1, q);   // Counter for valid elements beyond num_valid
    
    q.memset(d_left_write, 0, sizeof(int)).wait();
    q.memset(d_right_scan, 0, sizeof(int)).wait();
    
    // ===== SINGLE PASS: Scan entire array and record positions =====
    auto start_scan = std::chrono::high_resolution_clock::now();
    
    auto scan_event = q.submit([&](sycl::handler& h) {
        h.parallel_for<ScanAndRecordKernel>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            int i = idx[0];
            
            // Left half: Find invalid slots (gaps) in valid region [0, num_valid)
            if (i < num_valid && array[i].pointer == -1) {
                // Found a gap - atomically get position to record it
                sycl::atomic_ref<int, 
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space> atomic_left(*d_left_write);
                
                int pos = atomic_left.fetch_add(1);
                gap_positions[pos] = i;
            }
            
            // Right half: Find valid elements in invalid region [num_valid, n)
            if (i >= num_valid && array[i].pointer != -1) {
                // Found valid element that needs to move - atomically get position
                sycl::atomic_ref<int, 
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space> atomic_right(*d_right_scan);
                
                int pos = atomic_right.fetch_add(1);
                valid_positions[pos] = i;
            }
        });
    });
    scan_event.wait();
    
    auto end_scan = std::chrono::high_resolution_clock::now();
    
    // Get the counts back to host
    int gap_count, valid_count;
    q.memcpy(&gap_count, d_left_write, sizeof(int)).wait();
    q.memcpy(&valid_count, d_right_scan, sizeof(int)).wait();
    
    // Number of swaps needed is minimum of gaps and valid elements
    int swap_count = (gap_count < valid_count) ? gap_count : valid_count;
    
    // ===== PARALLEL SWAP: Match gaps with valid elements =====
    auto start_swap = std::chrono::high_resolution_clock::now();
    
    if (swap_count > 0) {
        auto swap_event = q.submit([&](sycl::handler& h) {
            h.parallel_for<SwapKernel>(sycl::range<1>(swap_count), [=](sycl::id<1> idx) {
                int i = idx[0];
                
                // Get the positions to swap
                int gap_idx = gap_positions[i];
                int valid_idx = valid_positions[i];
                
                // Perform the swap
                Element temp = array[gap_idx];
                array[gap_idx] = array[valid_idx];
                array[valid_idx] = temp;
            });
        });
        swap_event.wait();
    }
    
    auto end_swap = std::chrono::high_resolution_clock::now();
    
    // Cleanup atomic counters
    sycl::free(d_left_write, q);
    sycl::free(d_right_scan, q);
    
    // Calculate timing (only kernels)
    double scan_time = std::chrono::duration<double, std::milli>(end_scan - start_scan).count();
    double swap_time = std::chrono::duration<double, std::milli>(end_swap - start_swap).count();
    kernel_time_ms = scan_time + swap_time;
}

int main(int argc, char* argv[]) {
    
    // Parameters
    int n = 10000000;       // 10 million elements (default)
    double empty_ratio = 0.3;  // 30% empty spaces (default)
    int num_iterations = 1000; // Default 1000 iterations
    
    if (argc > 1) n = std::atoi(argv[1]);
    if (argc > 2) empty_ratio = std::atof(argv[2]);
    if (argc > 3) num_iterations = std::atoi(argv[3]);
    
    std::cout << "=== Dual-Pointer In-Place Array Compaction (SYCL) - Multiple Iterations ===" << std::endl;
    std::cout << "Array size: " << n << std::endl;
    std::cout << "Empty ratio: " << (empty_ratio * 100) << "%" << std::endl;
    std::cout << "Number of iterations: " << num_iterations << std::endl;
    
    // Create SYCL queue
    sycl::queue q{sycl::default_selector_v};
    
    std::cout << "\nRunning on: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Device type: ";
    if (q.get_device().is_cpu()) {
        std::cout << "CPU" << std::endl;
    } else if (q.get_device().is_gpu()) {
        std::cout << "GPU" << std::endl;
    } else {
        std::cout << "Other" << std::endl;
    }
    
    // Generate random test data on host (fixed for all iterations to isolate compaction timing)
    std::vector<Element> host_array(n);
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> coord_dist(0.0, 1000.0);
    std::uniform_real_distribution<double> empty_dist(0.0, 1.0);
    std::uniform_int_distribution<int> ptr_dist(1, 1000000);
    
    int actual_empty = 0;
    for (int i = 0; i < n; i++) {
        host_array[i].x = coord_dist(gen);
        host_array[i].y = coord_dist(gen);
        
        if (empty_dist(gen) < empty_ratio) {
            host_array[i].pointer = -1;
            actual_empty++;
        } else {
            host_array[i].pointer = ptr_dist(gen);
        }
    }
    
    int num_valid = n - actual_empty;
    std::cout << "\nActual empty count: " << actual_empty << std::endl;
    std::cout << "Valid elements: " << num_valid << std::endl;
    
    // Allocate device memory (once, reusable across iterations)
    Element* d_array = sycl::malloc_device<Element>(n, q);
    int* d_gap_positions = sycl::malloc_device<int>(actual_empty, q);  // Max possible gaps
    int* d_valid_positions = sycl::malloc_device<int>(actual_empty, q); // Max possible valid beyond
    
    // Initial copy to device
    q.memcpy(d_array, host_array.data(), n * sizeof(Element)).wait();
    
    // ===== MULTIPLE ITERATIONS: Reset data and compact each time =====
    auto start_iterations = std::chrono::high_resolution_clock::now();
    double total_kernel_time = 0.0;
    double total_reset_time = 0.0;
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        
        // Reset: Copy fresh data to device (simulates new input with empties each iteration)
        auto start_reset = std::chrono::high_resolution_clock::now();
        q.memcpy(d_array, host_array.data(), n * sizeof(Element)).wait();
        auto end_reset = std::chrono::high_resolution_clock::now();
        
        total_reset_time += std::chrono::duration<double, std::milli>(end_reset - start_reset).count();
        
        // Perform compaction (kernels only)
        double single_kernel_time;
        compactArrayDualPointer(q, d_array, d_gap_positions, d_valid_positions, 
                               n, num_valid, single_kernel_time);
        
        total_kernel_time += single_kernel_time;
        
        // Optional: Print progress every 100 iterations
        if ((iter + 1) % 100 == 0) {
            std::cout << "Completed " << (iter + 1) << "/" << num_iterations << " iterations" << std::endl;
        }
    }
    
    auto end_iterations = std::chrono::high_resolution_clock::now();
    
    // Total time including resets and kernels
    double total_compaction_time = std::chrono::duration<double, std::milli>(end_iterations - start_iterations).count();
    
    // Final copy back for verification (after last compaction)
    q.memcpy(host_array.data(), d_array, n * sizeof(Element)).wait();
    
    // ===== VERIFICATION (only on final iteration) =====
    std::cout << "\n=== Verification (Final Iteration) ===" << std::endl;
    int valid_count = 0;
    int invalid_in_range = 0;
    
    for (int i = 0; i < num_valid; i++) {
        if (host_array[i].pointer != -1) {
            valid_count++;
        } else {
            invalid_in_range++;
        }
    }
    
    std::cout << "Valid elements in compacted range [0, " << num_valid << "): " << valid_count << std::endl;
    std::cout << "Invalid elements in compacted range: " << invalid_in_range << std::endl;
    
    if (valid_count == num_valid && invalid_in_range == 0) {
        std::cout << "\n✓ COMPACTION SUCCESSFUL (Final Iteration)!" << std::endl;
    } else {
        std::cout << "\n✗ COMPACTION FAILED (Final Iteration)!" << std::endl;
        std::cout << "Expected " << num_valid << " valid elements, got " << valid_count << std::endl;
    }
    
    // Display sample of final compacted array
    std::cout << "\n=== Sample Output (first 10 valid elements - Final) ===" << std::endl;
    int displayed = 0;
    for (int i = 0; i < n && displayed < 10; i++) {
        if (host_array[i].pointer != -1) {
            std::cout << "array[" << i << "]: x=" << host_array[i].x 
                      << ", y=" << host_array[i].y 
                      << ", pointer=" << host_array[i].pointer << std::endl;
            displayed++;
        }
    }
    
    // ===== PERFORMANCE SUMMARY =====
    std::cout << "\n=== Performance Summary (All Iterations) ===" << std::endl;
    std::cout << "Total iterations: " << num_iterations << std::endl;
    std::cout << "Total elements processed: " << (n * num_iterations) << std::endl;
    std::cout << "Total data reset time (memcpy): " << total_reset_time << " ms" << std::endl;
    std::cout << "Total kernel time (compaction): " << total_kernel_time << " ms" << std::endl;
    std::cout << "Total compaction time (including resets): " << total_compaction_time << " ms" << std::endl;
    std::cout << "Average time per iteration (kernels only): " << (total_kernel_time / num_iterations) << " ms" << std::endl;
    std::cout << "Average time per iteration (total): " << (total_compaction_time / num_iterations) << " ms" << std::endl;
    std::cout << "Throughput (kernels only): " << ((n * num_iterations) / (total_kernel_time / 1000.0) / 1000000.0) << " million elements/sec" << std::endl;
    std::cout << "Throughput (total): " << ((n * num_iterations) / (total_compaction_time / 1000.0) / 1000000.0) << " million elements/sec" << std::endl;
    
    // Cleanup
    sycl::free(d_array, q);
    sycl::free(d_gap_positions, q);
    sycl::free(d_valid_positions, q);
    
    return 0;
}
