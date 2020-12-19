#pragma once

#include <cmath>
#include <cuda.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <vector>

namespace Boids {
void init_simulation(unsigned int N);
void step_simulation_naive(float dt);
void step_simulation_scattered_grid(float dt);
void step_simulation_coherent_grid(float dt);
void copy_boids_to_VBO(float* vbodptr_positions, float* vbodptr_velocities);

void end_simulation();
} // namespace Boids
