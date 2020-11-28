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
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

/*****************
 * Configuration *
 *****************/

/*! Block size used for CUDA kernel launch. */
constexpr unsigned int block_size = 128;

constexpr float rule1_distance = 60.0f;
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

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int* dev_particleArrayIndices; // What index in dev_pos and dev_velX represents
                               // this particle?
int* dev_particleGridIndices;  // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int* dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int* dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

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
  thrust::default_random_engine rng(hash(static_cast<int>(index * time)));
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

  // LOOK-2.1 computing grid params
  gridCellWidth =
      2.0f * std::max(std::max(rule1_distance, rule2_distance), rule3_distance);
  const int half_side_count = static_cast<int>(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * half_side_count;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * half_side_count;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

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
    if (i == i_self) {
      continue;
    }

    const float distance = glm::distance(pos[i], pos[i_self]);
    if (distance < rule1_distance) {
      perceived_center += pos[i];
      ++rule1_neighbor_count;
    }

    if (distance < rule2_distance) {
      c -= (pos[i] - pos[i_self]);
    }

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
  if (index >= N) {
    return;
  }

  for (auto i = 0u; i < N; ++i) {
    // Compute a new velocity based on pos and vel1
    auto new_velocity = compute_velocity_change(N, i, pos, vel1);
    // Clamp the speed
    if (dot(new_velocity, new_velocity) > max_speed * max_speed) {
      new_velocity = glm::normalize(new_velocity);
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
  if (index >= N) {
    return;
  }
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
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution)
{
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution, glm::vec3 gridMin,
                                   float inverseCellWidth, glm::vec3* pos,
                                   int* indices, int* gridIndices)
{
  // TODO-2.1
  // - Label each boid with the index of its grid cell.
  // - Set up a parallel array of integer indices as pointers to the actual
  //   boid data in pos and vel1/vel2
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int* intBuffer, int value)
{
  const auto index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int* particleGridIndices,
                                         int* gridCellStartIndices,
                                         int* gridCellEndIndices)
{
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
}

__global__ void kernUpdateVelNeighborSearchScattered(
    int N, int gridResolution, glm::vec3 gridMin, float inverseCellWidth,
    float cellWidth, int* gridCellStartIndices, int* gridCellEndIndices,
    int* particleArrayIndices, glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2)
{
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
}

__global__ void kernUpdateVelNeighborSearchCoherent(
    int N, int gridResolution, glm::vec3 gridMin, float inverseCellWidth,
    float cellWidth, int* gridCellStartIndices, int* gridCellEndIndices,
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
void Boids::stepSimulationNaive(float dt)
{
  dim3 fullBlocksPerGrid((objects_count + block_size - 1) / block_size);

  kern_update_pos<<<fullBlocksPerGrid, block_size>>>(objects_count, dt, dev_pos,
                                                     dev_vel1);
  kern_update_velocity_brute_force<<<fullBlocksPerGrid, block_size>>>(
      objects_count, dev_pos, dev_vel1, dev_vel2);

  // ping-pong the velocity buffers
  std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt)
{
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed
}

void Boids::stepSimulationCoherentGrid(float dt)
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
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
}

void Boids::unitTest()
{
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int* dev_intKeys = nullptr;
  int* dev_intValues = nullptr;
  int N = 10;

  std::unique_ptr<int[]> intKeys{new int[N]};
  std::unique_ptr<int[]> intValues{new int[N]};

  intKeys[0] = 0;
  intValues[0] = 0;
  intKeys[1] = 1;
  intValues[1] = 1;
  intKeys[2] = 0;
  intValues[2] = 2;
  intKeys[3] = 3;
  intValues[3] = 3;
  intKeys[4] = 0;
  intValues[4] = 4;
  intKeys[5] = 2;
  intValues[5] = 5;
  intKeys[6] = 2;
  intValues[6] = 6;
  intKeys[7] = 0;
  intValues[7] = 7;
  intKeys[8] = 5;
  intValues[8] = 8;
  intKeys[9] = 6;
  intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + block_size - 1) / block_size);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N,
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N,
             cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N,
             cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
