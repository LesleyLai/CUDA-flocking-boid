#define GLM_FORCE_CUDA
#include "kernel.h"
#include "utilityCore.hpp"
#include <cmath>
#include <cuda.h>
#include <glm/glm.hpp>
#include <stdio.h>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef imin
#define imin(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAError(const char* msg, int line = -1)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) { fprintf(stderr, "Line %d: ", line); }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/*****************
 * Configuration *
 *****************/

/*! Block size used for CUDA kernel launch. */
constexpr unsigned int block_size = 128;

constexpr float rule1_distance = 30.0f;
constexpr float rule2_distance = 3.0f;
constexpr float rule3_distance = 5.0f;

constexpr float rule1_scale = 0.01f;
constexpr float rule2_scale = 0.1f;
constexpr float rule3_scale = 0.1f;

constexpr float max_speed = 1.0f;

/*! Size of the starting area in simulation space. */
constexpr float scene_scale = 100.0f;

/***********************************************
 * Kernel state (pointers are device pointers) *
 ***********************************************/

unsigned int objects_count = 0;
static constexpr dim3 threads_per_block(block_size);

// These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3* dev_pos = nullptr;
glm::vec3* dev_vel1 = nullptr;
glm::vec3* dev_vel2 = nullptr;

// For efficient sorting and the uniform grid. These should always be parallel.
unsigned int* dev_particle_array_indices =
    nullptr; // index in dev_pos and dev_velX for each boid
unsigned int* dev_particle_grid_indices = nullptr; // grid index of each boid

// What part of dev_particle_array_indices belongs to this cell?
int* dev_grid_cell_start_indices = nullptr;
int* dev_grid_cell_end_indices = nullptr;

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
// computing grid params

constexpr float grid_cell_width =
    2.0f * std::max(std::max(rule1_distance, rule2_distance), rule3_distance);
constexpr int half_side_count =
    static_cast<int>(scene_scale / grid_cell_width) + 1;
constexpr int grid_side_count = 2 * half_side_count;
constexpr int grid_cell_count =
    grid_side_count * grid_side_count * grid_side_count;
constexpr float grid_inverse_cell_width = 1.0f / grid_cell_width;
constexpr glm::vec3 grid_minimum{-grid_cell_width * half_side_count,
                                 -grid_cell_width* half_side_count,
                                 -grid_cell_width* half_side_count};

/******************
 * initSimulation *
 ******************/

__host__ __device__ auto hash(unsigned int a) -> unsigned int
{
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

__host__ __device__ auto generate_random_vec3(float time, unsigned int index)
    -> glm::vec3
{
  thrust::default_random_engine rng(
      hash(static_cast<int>(static_cast<float>(index) * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3(unitDistrib(rng), unitDistrib(rng), unitDistrib(rng));
}

__global__ void kern_generate_random_pos_array(int time, int N, glm::vec3* arr,
                                               float scale)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generate_random_vec3(time, index);
    arr[index] = scale * rand;
  }
}

/**
 * Initialize memory, update some globals
 */
void Boids::init_simulation(unsigned int N)
{
  objects_count = N;
  dim3 fullBlocksPerGrid((N + block_size - 1) / block_size);

  cudaMalloc(reinterpret_cast<void**>(&dev_pos), N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc(reinterpret_cast<void**>(&dev_vel1), N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc(reinterpret_cast<void**>(&dev_vel2), N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  kern_generate_random_pos_array<<<fullBlocksPerGrid, block_size>>>(
      1, objects_count, dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernel_generate_random_pos_array failed!");

  cudaMalloc(reinterpret_cast<void**>(&dev_particle_array_indices),
             N * sizeof(unsigned int));
  checkCUDAErrorWithLine("cudaMalloc dev_particle_array_indices failed!");
  cudaMalloc(reinterpret_cast<void**>(&dev_particle_grid_indices),
             N * sizeof(unsigned int));
  checkCUDAErrorWithLine("cudaMalloc dev_particle_grid_indices failed!");
  cudaMalloc(reinterpret_cast<void**>(&dev_grid_cell_start_indices),
             grid_cell_count * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_grid_cell_start_indices failed!");
  cudaMalloc(reinterpret_cast<void**>(&dev_grid_cell_end_indices),
             grid_cell_count * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_grid_cell_start_indices failed!");
  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaDeviceSynchronize();
}

/******************
 * copy_boids_to_VBO *
 ******************/

/**
 * Copy the boid positions into the VBO so that they can be drawn by OpenGL.
 */
__global__ void kern_copy_positions_to_VBO(int N, glm::vec3* pos, float* vbo,
                                           float s_scale)
{
  const auto index = threadIdx.x + (blockIdx.x * blockDim.x);

  const float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kern_copy_velocities_to_VBO(int N, glm::vec3* vel, float* vbo,
                                            float s_scale)
{
  const auto index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

void Boids::copy_boids_to_VBO(float* vbodptr_positions,
                              float* vbodptr_velocities)
{
  dim3 fullBlocksPerGrid((objects_count + block_size - 1) / block_size);

  kern_copy_positions_to_VBO<<<fullBlocksPerGrid, block_size>>>(
      objects_count, dev_pos, vbodptr_positions, scene_scale);
  kern_copy_velocities_to_VBO<<<fullBlocksPerGrid, block_size>>>(
      objects_count, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copy_boids_to_VBO failed!");

  cudaDeviceSynchronize();
}

/******************
 * stepSimulation *
 ******************/

/**
 * Compute the new velocity on the body with index `i_self` due to the `N` boids
 * in the `pos` and `vel` arrays.
 */
__device__ auto compute_velocity_change(unsigned int N, unsigned int i_self,
                                        const glm::vec3* pos,
                                        const glm::vec3* vel) -> glm::vec3
{
  glm::vec3 perceived_center{}, c{}, perceived_velocity{};
  int rule1_neighbor_count = 0, rule3_neighbor_count = 0;
  for (auto i = 0u; i < N; ++i) {
    if (i == i_self) { continue; }

    const float distance = glm::distance(pos[i], pos[i_self]);
    if (distance < rule1_distance) {
      perceived_center += pos[i];
      ++rule1_neighbor_count;
    }

    if (distance < rule2_distance) { c -= (pos[i] - pos[i_self]); }

    if (distance < rule3_distance) {
      perceived_velocity += vel[i];
      ++rule3_neighbor_count;
    }
  }

  glm::vec3 result = vel[i_self];
  if (rule1_neighbor_count != 0) {
    perceived_center /= rule1_neighbor_count;
    result += (perceived_center - pos[i_self]) * rule1_scale;
  }
  result += c * rule2_scale;
  if (rule3_neighbor_count != 0) {
    perceived_velocity /= rule3_neighbor_count;
    result += perceived_velocity * rule3_scale;
  }
  return result;
}

__global__ void kern_update_velocity_brute_force(unsigned int N, glm::vec3* pos,
                                                 glm::vec3* vel1,
                                                 glm::vec3* vel2)
{
  const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) { return; }

  for (auto i = 0u; i < N; ++i) {
    // Compute a new velocity based on pos and vel1
    auto new_velocity = compute_velocity_change(N, i, pos, vel1);
    // Clamp the speed
    if (dot(new_velocity, new_velocity) > max_speed * max_speed) {
      new_velocity = glm::normalize(new_velocity) * max_speed;
    }
    // Record the new velocity into vel2. Question: why NOT vel1?
    vel2[i] = new_velocity;
  }
}

/**
 * For each of the `N` bodies, update its position based on its current
 * velocity.
 */
__global__ void kern_update_pos(unsigned int N, float dt, glm::vec3* pos,
                                glm::vec3* vel)
{
  // Update position by velocity
  const auto index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) { return; }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ auto grid_index_3D_to_1D(unsigned int x, unsigned int y,
                                    unsigned int z) -> unsigned int
{
  return x + y * grid_side_count + z * grid_side_count * grid_side_count;
}

__global__ void kern_compute_indices(unsigned int N, glm::vec3 grid_min,
                                     glm::vec3* pos, unsigned int* indices,
                                     unsigned int* grid_indices)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) return;

  // - Label each boid with the index of its grid cell.
  glm::tvec3<unsigned int> grid_3d_indices =
      (pos[index] - grid_min) * grid_inverse_cell_width;

  grid_indices[index] = grid_index_3D_to_1D(
      grid_3d_indices.x, grid_3d_indices.y, grid_3d_indices.z);

  // - Set up a parallel array of integer indices as pointers to the actual
  //   boid data in pos and vel1/vel2
  indices[index] = index;
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kern_reset_int_buffer(unsigned int N, int* intBuffer, int value)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) { intBuffer[index] = value; }
}

// TODO: better way to do this?
__global__ void kern_identify_cell_start_end(
    unsigned int N, unsigned int* particle_grid_indices,
    int* grid_cell_start_indices, int* grid_cell_end_indices)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) return;

  const auto current_grid_index = particle_grid_indices[index];

  if (index == 0) {
    grid_cell_start_indices[current_grid_index] = 0;
    return;
  }

  const auto prev_grid_index = particle_grid_indices[index - 1];
  if (current_grid_index != prev_grid_index) {
    grid_cell_end_indices[prev_grid_index] = static_cast<int>(index - 1);
    grid_cell_start_indices[current_grid_index] = index;

    if (index == N - 1) { grid_cell_end_indices[current_grid_index] = index; }
  }
}

__global__ void kern_update_vel_neighbor_search_scattered(
    unsigned int N, int* grid_cell_start_indices, int* grid_cell_end_indices,
    unsigned int* particle_array_indices, glm::vec3* pos, glm::vec3* vel1,
    glm::vec3* vel2)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) return;

  // - Identify the grid cell that this particle is in
  const auto x = static_cast<int>((pos[index].x - grid_minimum.x) *
                                  grid_inverse_cell_width);
  const auto y = static_cast<int>((pos[index].y - grid_minimum.y) *
                                  grid_inverse_cell_width);
  const auto z = static_cast<int>((pos[index].z - grid_minimum.z) *
                                  grid_inverse_cell_width);
  // const auto grid_index = grid_index_3D_to_1D(x, y, z);

  glm::vec3 perceived_center{}, c{}, perceived_velocity{};
  int rule1_neighbor_count = 0, rule3_neighbor_count = 0;

  // - Identify which cells may contain neighbors. This isn't always 8.
  // TODO: checks
  for (int xx = x - 1; xx < x + 1; ++xx) {
    if (xx < 0 || xx >= grid_side_count) continue;
    for (int yy = y - 1; yy < y + 1; ++yy) {
      if (yy < 0 || yy >= grid_side_count) continue;
      for (int zz = z - 1; zz < z + 1; ++zz) {
        if (zz < 0 || zz >= grid_side_count) continue;
        const auto neighbor_grid_index = grid_index_3D_to_1D(xx, yy, zz);

        // - For each cell, read the start/end indices
        const auto neighbor_start =
            grid_cell_start_indices[neighbor_grid_index];
        const auto neighbor_end = grid_cell_end_indices[neighbor_grid_index];

        if (neighbor_start == -1) continue;

        // - Access each boid in the cell and compute velocity change from
        //   the boids rules, if this boid is within the neighborhood distance.
        for (int i = neighbor_start; i <= neighbor_end; ++i) {
          const auto neighbor_boid_index = particle_array_indices[i];
          if (neighbor_boid_index == index) continue;

          const auto neighbor_pos = pos[neighbor_boid_index];
          const auto neighbor_vel = vel1[neighbor_boid_index];
          const float distance = glm::distance(neighbor_pos, pos[index]);
          if (distance < rule1_distance) {
            perceived_center += neighbor_pos;
            ++rule1_neighbor_count;
          }

          if (distance < rule2_distance) { c -= (neighbor_pos - pos[index]); }

          if (distance < rule3_distance) {
            perceived_velocity += neighbor_vel;
            ++rule3_neighbor_count;
          }
        }
      }
    }
  }

  glm::vec3 new_velocity = vel1[index];
  if (rule1_neighbor_count != 0) {
    perceived_center /= rule1_neighbor_count;
    new_velocity += (perceived_center - pos[index]) * rule1_scale;
  }
  new_velocity += c * rule2_scale;
  if (rule3_neighbor_count != 0) {
    perceived_velocity /= rule3_neighbor_count;
    new_velocity += perceived_velocity * rule3_scale;
  }

  // - Clamp the speed change before putting the new speed in vel2
  if (dot(new_velocity, new_velocity) > max_speed * max_speed) {
    new_velocity = glm::normalize(new_velocity) * max_speed;
  }
  vel2[index] = new_velocity;
}

__global__ void kern_update_vel_neighbor_search_coherent(
    unsigned int N, int* gridCellStartIndices, int* gridCellEndIndices,
    glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2)
{
  // TODO-2.3 - This should be very similar to
  // kernUpdateVelNeighborSearchScattered, except with one less level of
  // indirection. This should expect gridCellStartIndices and gridCellEndIndices
  // to refer directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
}

/**
 * Step the entire N-body simulation by `dt` seconds.
 */
void Boids::step_simulation_naive(float dt)
{
  const dim3 full_blocks_per_grid((objects_count + block_size - 1) /
                                  block_size);

  kern_update_pos<<<full_blocks_per_grid, block_size>>>(objects_count, dt,
                                                        dev_pos, dev_vel1);
  kern_update_velocity_brute_force<<<full_blocks_per_grid, block_size>>>(
      objects_count, dev_pos, dev_vel1, dev_vel2);

  // ping-pong the velocity buffers
  std::swap(dev_vel1, dev_vel2);
}

void Boids::step_simulation_scattered_grid(float dt)
{
  // TODO-2.1

  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  const dim3 full_blocks_per_grid((objects_count + block_size - 1) /
                                  block_size);
  const dim3 grid_size((grid_cell_count + block_size - 1) / block_size);
  kern_compute_indices<<<full_blocks_per_grid, block_size>>>(
      objects_count, grid_minimum, dev_pos, dev_particle_array_indices,
      dev_particle_grid_indices);

  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  auto thrust_indices = thrust::device_pointer_cast(dev_particle_array_indices);
  auto thrust_grid_indices =
      thrust::device_pointer_cast(dev_particle_grid_indices);
  thrust::sort_by_key(thrust_grid_indices, thrust_grid_indices + objects_count,
                      thrust_indices);

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  kern_reset_int_buffer<<<grid_size, block_size>>>(
      grid_cell_count, dev_grid_cell_start_indices, -1);
  kern_reset_int_buffer<<<grid_size, block_size>>>(
      grid_cell_count, dev_grid_cell_end_indices, -1);
  kern_identify_cell_start_end<<<full_blocks_per_grid, block_size>>>(
      objects_count, dev_particle_grid_indices, dev_grid_cell_start_indices,
      dev_grid_cell_end_indices);

  //  std::cout << "array indices\n";
  //  thrust::copy(thrust_indices, thrust_indices + objects_count,
  //               std::ostream_iterator<float>(std::cout, " "));
  //  std::cout << '\n';
  //  std::cout << "grid indices\n";
  //  thrust::copy(thrust_grid_indices, thrust_grid_indices + objects_count,
  //               std::ostream_iterator<float>(std::cout, " "));
  //  std::cout << '\n';

  // - Perform velocity updates using neighbor search
  kern_update_vel_neighbor_search_scattered<<<full_blocks_per_grid,
                                              block_size>>>(
      objects_count, dev_grid_cell_start_indices, dev_grid_cell_end_indices,
      dev_particle_array_indices, dev_pos, dev_vel1, dev_vel2);

  // - Update positions
  kern_update_pos<<<full_blocks_per_grid, block_size>>>(objects_count, dt,
                                                        dev_pos, dev_vel2);
  // - Ping-pong buffers as needed
  std::swap(dev_vel1, dev_vel2);
}

void Boids::step_simulation_coherent_grid(float dt)
{
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::end_simulation()
{
  cudaFree(dev_grid_cell_start_indices);
  cudaFree(dev_grid_cell_end_indices);
  cudaFree(dev_particle_array_indices);
  cudaFree(dev_particle_grid_indices);

  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);
}
