#include <metal_stdlib>
using namespace metal;

struct KernelParams {
	uint num_sites;
	uint num_updated_nodes;
	uint leaves_end;
	uint scale_ln;
	float scale;
	float inv_scale;
	float2 _pad;
	float4 frequencies;
};

static inline uint projection_idx(uint node, uint site, uint sub, uint num_sites) {
	return (node * num_sites + site) * 4 + sub;
}

// Build one component of the 4-state projection from the ambiguity mask.
static inline float calc_leaf_projection(uchar leaf, float4 row) {
	float out = 0.0f;
	if (leaf & 0b0001) out += row.x;
	if (leaf & 0b0010) out += row.y;
	if (leaf & 0b0100) out += row.z;
	if (leaf & 0b1000) out += row.w;
	return out;
}

// Four neighboring threads handle the four state components of one pattern.
kernel void propose_kernel(
	const device uchar* leaves [[buffer(0)]],
	device float* projections [[buffer(1)]],
	device uchar* scales [[buffer(2)]],
	device uint* scale_sums [[buffer(3)]],
	const device uint* nodes [[buffer(4)]],
	const device uint* children [[buffer(5)]],
	const device float4* transitions [[buffer(6)]],
	device float* likelihoods [[buffer(7)]],
	const device KernelParams* params_buf [[buffer(8)]],
	uint gid [[thread_position_in_grid]],
	uint lane [[thread_index_in_simdgroup]]
) {
	const device KernelParams& params = params_buf[0];
	uint num_sites = params.num_sites;
	uint num_updated = params.num_updated_nodes;
	uint leaves_end = params.leaves_end;
	uint scale_ln = params.scale_ln;
	float scale = params.scale;
	float inv_scale = params.inv_scale;
	float4 frequencies = params.frequencies;

	uint site = gid >> 2;
	uint sub = gid & 3;

	if (site >= num_sites) {
		return;
	}

	uint scale_sum = scale_sums[site];

	for (uint i = 0; i < leaves_end; i++) {
		uint node = nodes[i];
		uchar leaf = leaves[node * num_sites + site];
		uint tbase = i * 4;

		projections[projection_idx(node, site, sub, num_sites)] =
			calc_leaf_projection(leaf, transitions[tbase + sub]);
	}

	threadgroup_barrier(mem_flags::mem_device);

	for (uint i = leaves_end; i < num_updated; i++) {
		uint node = nodes[i];
		uint left = children[(i - leaves_end) * 2 + 0];
		uint right = children[(i - leaves_end) * 2 + 1];

		uint left_idx = projection_idx(left, site, sub, num_sites);
		uint right_idx = projection_idx(right, site, sub, num_sites);
		float local = projections[left_idx] * projections[right_idx];

		uint base = lane - sub;
		float4 like = float4(
			simd_shuffle(local, base + 0),
			simd_shuffle(local, base + 1),
			simd_shuffle(local, base + 2),
			simd_shuffle(local, base + 3)
		);
		bool should_scale = like.x < scale
			&& like.y < scale
			&& like.z < scale
			&& like.w < scale;

		uint proj_idx = node * num_sites + site;
		if (should_scale) {
			local *= inv_scale;
			like *= inv_scale;
		}

		if (sub == 0) {
			uchar old_scale = scales[proj_idx];
			if (should_scale) {
				if (old_scale == 0) {
					scale_sum += scale_ln;
					scales[proj_idx] = 1;
				}
			} else if (old_scale != 0) {
				scale_sum -= scale_ln;
				scales[proj_idx] = 0;
			}
		}

		uint tbase = i * 4;
		projections[projection_idx(node, site, sub, num_sites)] =
			dot(transitions[tbase + sub], like);

		threadgroup_barrier(mem_flags::mem_device);
	}

	uint root = nodes[num_updated];
	uint left = children[(num_updated - leaves_end) * 2 + 0];
	uint right = children[(num_updated - leaves_end) * 2 + 1];

	float left_projection =
		projections[projection_idx(left, site, sub, num_sites)];
	float right_projection =
		projections[projection_idx(right, site, sub, num_sites)];
	float local = left_projection * right_projection * frequencies[sub];

	if (sub == 0) {
		uint base = lane;
		float sum = local
			+ simd_shuffle(local, base + 1)
			+ simd_shuffle(local, base + 2)
			+ simd_shuffle(local, base + 3);
		likelihoods[site] = log(sum);

		uint root_idx = root * num_sites + site;
		if (scales[root_idx] != 0) {
			scales[root_idx] = 0;
			scale_sum -= scale_ln;
		}

		scale_sums[site] = scale_sum;
	}
}
