#ifndef SYCL_HASHMAP_HPP
#define SYCL_HASHMAP_HPP

#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>
#include <iostream>
#include "point.hpp"
namespace sycl_hashmap {

// A constant to denote an empty key (all valid grid locations are assumed non-negative).
constexpr int EMPTY_KEY = -1;

// ---------------------------------------------------------------------------------
// Device view structure
// This plain-vanilla structure contains only the information needed by a kernel.
// It is trivially copyable so that it can be captured by the kernel lambda.
struct DeviceView {
  int *keys;
  int *values;
  int capacity;
  int probing_length; // Dynamic probing length passed from the host.
  int *key_idx;
  int *delete_idx;
  // Insert a (key,value) pair into the hash map using linear probing.
  inline int pop_from_deleted(int grid_key) const{
    int head = delete_idx[grid_key];
    if(head == EMPTY_KEY)return EMPTY_KEY;
    int next = values[head];
    delete_idx[grid_key] = next;
    return head;
  }
  inline int pop_from_free_list (int grid_key) const {
    // Atomic reference to the head of the freelist for this grid_key
    auto atomic_head = sycl::atomic_ref<int,
                        sycl::memory_order::acq_rel,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>(
        delete_idx[grid_key]);

    int head, next;
    do {
        head = atomic_head.load();
        if (head == EMPTY_KEY) {
            // No deleted slots to reuse
            return EMPTY_KEY;
        }
        // 'values[head]' is the "next" pointer in the freelist
         auto atomic_next = sycl::atomic_ref<int,
                            sycl::memory_order::acq_rel,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
            values[head]);
        next = atomic_next.load();
    } while (!atomic_head.compare_exchange_strong(head, next));

    // Successfully popped 'head' from the list
    return head;
  }
  inline void cleanup(int grid_key, Point* points_device) const{
    int base = grid_key*probing_length;
    int k_idx = key_idx[grid_key] - 1;

    if(k_idx < 0){
      return;
    }

    int free_space = pop_from_deleted(grid_key);
    

    while(free_space != EMPTY_KEY && k_idx != -1)
    {
      while(k_idx >= 0 && keys[k_idx + base] == 0)
      {
        k_idx--;
      }
      if(k_idx < 0){
        values[free_space] = delete_idx[grid_key];
        delete_idx[grid_key] = free_space;
        break;
      }

      if(free_space >= k_idx + base)
      {
        keys[free_space] = -1;
        values[free_space] = -1;
      }else
      {
        keys[k_idx + base] = -1;
        keys[free_space] = 1;
        values[free_space] = values[k_idx + base];
        values[k_idx + base] = -1;
        k_idx--;
        points_device[values[free_space]].pointer = free_space;
      }
      free_space = pop_from_deleted(grid_key);
    }
    key_idx[grid_key] = k_idx + 1;
  }

  inline void cleanup_clear_freelist(int grid_key, Point* points_device) const {
    int base = grid_key * probing_length;
    int k_idx = key_idx[grid_key] - 1;
    
    // If there are occupied slots, do normal cleanup (or the compaction you prefer)
    if (k_idx >= 0) {
      // Option A: run your existing compaction code (move tail elements into free slots).
      // Option B: if you prefer to clear freelist only when the bucket truly becomes empty,
      // you can return here and let other logic handle compaction. For now, we keep original behavior.
      // (If you want compaction here, reuse your existing cleanup body.)
      return;
    }

    // Bucket is empty — clear the whole free list for this bucket.
    int head = delete_idx[grid_key];    // safe non-atomic because cleanup runs single-threaded
    while (head != EMPTY_KEY) {
      int next = values[head];          // values[head] stores next pointer in freelist
      // reset slot to never-used state
      keys[head] = EMPTY_KEY;           // mark truly empty
      values[head] = -1;                // clear next / value
      // if you store any back-pointers in points_device, clear or update them if necessary:
      // e.g., points_device[...] no longer points here — but if bucket is empty, probably not needed.
      head = next;
    }

    // reset freelist head and key index
    delete_idx[grid_key] = EMPTY_KEY;
    key_idx[grid_key] = 0;
  }

  inline int insert(int key, int value,int* NUM_Points) const {
      //int grid_key = key / probing_length;
	//int base = key * probing_length;
	//int end = base + probing_length;
      //First try to reuse a deleted slot
      int idx = pop_from_free_list(key);
      while (idx != EMPTY_KEY) {
          // Try to claim the popped slot. Expect DELETED_MARK (0).
          auto atomic_slot_key = sycl::atomic_ref<int,
                                      sycl::memory_order::acq_rel,
                                      sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>(
                                          keys[idx]);
          int expected = /* DELETED_MARK */ 0;
          // If we succeed, we have exclusive right to initialize this slot.
          if (atomic_slot_key.compare_exchange_strong(expected, /*OCCUPIED_MARK*/1)) {
              // initialize value (next pointer is no longer freelist)
              auto atomic_val = sycl::atomic_ref<int,
                                      sycl::memory_order::acq_rel,
                                      sycl::memory_scope::device,
                                      sycl::access::address_space::global_space>(
                                          values[idx]);
              atomic_val.store(value);

              // update bucket metadata (key_idx) if necessary atomically:
              // e.g., advance key_idx with fetch_add(1) or set to max position;
              // here we simply ensure key_idx at least accounts for this slot if you track positions
              // NOTE: adapt to your key_idx semantics.
              // atomic_kidx.fetch_add(1);  // example

              return idx;
          }
          // somebody else raced and took this slot -> try next freed slot
          idx = pop_from_free_list(key);
      }


      // Otherwise fall back to probing
      int hash = key * probing_length;
      int k_idx = key_idx[key];
      for (int i = k_idx; i < probing_length; i++) {
          int prob_idx = hash + i;
          if (prob_idx >= capacity) return -1;

          auto atomic_key = sycl::atomic_ref<int,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space>(
              keys[prob_idx]);
          int expected = EMPTY_KEY;
          if (atomic_key.compare_exchange_strong(expected, key)) {
              keys[prob_idx]  = 1;
              values[prob_idx] = value;
              // auto atomic_kidx = sycl::atomic_ref<int,
              //       sycl::memory_order::relaxed,
              //       sycl::memory_scope::device,
              //       sycl::access::address_space::global_space>(key_idx[key]);

              // int old_value = atomic_kidx.fetch_add(1);

              key_idx[key] = i+1;
              //(*NUM_Points)++;
              return prob_idx;
          }
      }
      return -1;
  }

  inline void deletion(int key,int* NUM_Points) const{
      int grid_key = key / probing_length;

      // Free the slot
      auto akeys = sycl::atomic_ref<int,
             sycl::memory_order::acq_rel,
             sycl::memory_scope::device,
             sycl::access::address_space::global_space>(keys[key]);
      akeys.store(0);

      // Atomic reference to the bucket head
      auto atomic_head = sycl::atomic_ref<int,
                          sycl::memory_order::acq_rel,
                          sycl::memory_scope::device,
                          sycl::access::address_space::global_space>(
          delete_idx[grid_key]);

      int old_head;
      do {
          // Read current head
          old_head = atomic_head.load();

          // Link this slot to the old head
          auto atomic_next = sycl::atomic_ref<int,
                            sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
            values[key]);
        atomic_next.store(old_head);
          //(*NUM_Points)--;
          // Try to set this key as the new head
      } while (!atomic_head.compare_exchange_strong(old_head, key));
  }
  
  inline int* retrieve(int key) const{
    int idx = key * probing_length;
    if(idx >= capacity)
      return NULL;
    int *v = values + idx;
    return v;
  }
};

// ---------------------------------------------------------------------------------
// Host-based hash map class.
// This class allocates USM for the keys and values arrays and provides host-side access
// (e.g. retrieval and printing). It also provides a device view for use by kernels.
class SyclHashMap {
public:
  // Constructor.
  //   capacity      : total number of available slots.
  //   queue         : a SYCL queue used (on the host) for USM allocation.
  //   probing_length: the number of probing iterations (dynamic value).
  SyclHashMap(int capacity, sycl::queue &queue, int probing_length,int key_range)
      : capacity_{capacity}, probing_length_{probing_length} {
    keys_ = sycl::malloc_device<int>(capacity_, queue);
    values_ = sycl::malloc_device<int>(capacity_, queue);
    key_idx_ = sycl::malloc_device<int>(key_range,queue);
    delete_idx_ = sycl::malloc_device<int>(key_range,queue);
    // Initialize all slots to EMPTY_KEY.
    // for (int i = 0; i < capacity_; ++i) {
    //   keys_[i] = EMPTY_KEY;
    //   values_[i] = -1;
    // }
    // for(int i = 0;i < key_range;i++)
    // {
    //     key_idx_[i] = 0;
    // }
#ifndef __SYCL_DEVICE_ONLY__
    hostQueue_ = &queue;
#endif
  }

  // Destructor.
  ~SyclHashMap() {
#ifndef __SYCL_DEVICE_ONLY__
    if (hostQueue_) {
      sycl::free(keys_, *hostQueue_);
      sycl::free(values_, *hostQueue_);
      sycl::free(key_idx_, *hostQueue_);
      sycl::free(delete_idx_,*hostQueue_);
    }
#endif
  }

  // Create a device view that contains only the information needed by kernels.
  DeviceView device_view() const {
    return DeviceView{keys_, values_, capacity_, probing_length_,key_idx_,delete_idx_};
  }

  // Host-side retrieval function.
  // Scans through up to probing_length slots (starting from the hash index)
  // and collects values that match the key.
  inline std::vector<int> retrieve(int key) const {
    std::vector<int> result;
    int hash = key * probing_length_;
    for (int i = 0; i < probing_length_; i++) {
      int idx = hash + i;
      if (keys_[idx] != key)
        break;
      if (keys_[idx] == key)
        result.push_back(values_[idx]);
    }
    return result;
  }

  // Debug print: output all non-empty slots.
  inline void print_all() const {
    std::cout << "HashMap contents:\n";
    int count = 0;
    for (int i = 0; i < capacity_; i++) {
      if (keys_[i] != EMPTY_KEY){
        //std::cout << "Key : " << keys_[i] << " Value : " << values_[i] << "\n";
        count++;
      }
    }
    std::cout << "Number of Elements = " << count << "\n"; 
  }

private:
  // USM pointers for the key and value arrays.
  int *keys_;
  int *values_;
  int *key_idx_;
  int capacity_;
  int probing_length_; // stored dynamic probing length
  int *delete_idx_;
#ifndef __SYCL_DEVICE_ONLY__
  // The host queue pointer is stored only on the host.
  sycl::queue* hostQueue_;
#endif
};

} // namespace sycl_hashmap

#endif // SYCL_HASHMAP_HPP

