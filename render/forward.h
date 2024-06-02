#pragma once
#include "auxiliary.h"

struct DepthPair {
	int idx;
	float depth;	
};

void  PreRender(
	const float* xyz, 
	const float* rotation,
	const float* scaling,
	const float* features,

	const float near,
	const int N, 
	const int width,
	const int height,
	const float* viewMatrix, 
	const float* viewProjMatrix,
	const float* cameraCenter,
	float focalX, float focalY,
	float tanFovX, float tanFovY);

struct TileDepth {
	int gaussianId;
	uint64_t key;

	__host__ __device__
		bool operator<(const TileDepth& other) const {
			return key < other.key;
	}
};