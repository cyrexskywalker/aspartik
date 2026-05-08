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

// Build the 4-state projection for one leaf/site pair from the ambiguity mask.
static inline float4 calc_leaf_projection(uchar leaf, float4 t0, float4 t1, float4 t2, float4 t3) {
	float4 out = float4(0.0f);
	if (leaf & 0b0001) out += float4(t0.x, t1.x, t2.x, t3.x);
	if (leaf & 0b0010) out += float4(t0.y, t1.y, t2.y, t3.y);
	if (leaf & 0b0100) out += float4(t0.z, t1.z, t2.z, t3.z);
	if (leaf & 0b1000) out += float4(t0.w, t1.w, t2.w, t3.w);
	return out;
}

// One thread handles one alignment site and walks the update list in order.
kernel void propose_kernel(
	const device uchar* leaves [[buffer(0)]],
	device float4* projections [[buffer(1)]],
	device uchar* scales [[buffer(2)]],
	device uint* scale_sums [[buffer(3)]],
	const device uint* nodes [[buffer(4)]],
	const device uint* children [[buffer(5)]],
	const device float4* transitions [[buffer(6)]],
	device float* likelihoods [[buffer(7)]],
	const device KernelParams* params_buf [[buffer(8)]],
	uint gid [[thread_position_in_grid]]
) {
	const device KernelParams& params = params_buf[0];
	uint num_sites = params.num_sites;
	uint num_updated = params.num_updated_nodes;
	uint leaves_end = params.leaves_end;
	uint scale_ln = params.scale_ln;
	float scale = params.scale;
	float inv_scale = params.inv_scale;
	float4 frequencies = params.frequencies;

	if (gid >= num_sites) {
		return;
	}

	uint site = gid;
	uint scale_sum = scale_sums[site];

	for (uint i = 0; i < leaves_end; i++) {
		uint node = nodes[i];
		uchar leaf = leaves[node * num_sites + site];
		uint tbase = i * 4;

		float4 projection = calc_leaf_projection(
			leaf,
			transitions[tbase + 0],
			transitions[tbase + 1],
			transitions[tbase + 2],
			transitions[tbase + 3]
		);
		projections[node * num_sites + site] = projection;
	}

	for (uint i = leaves_end; i < num_updated; i++) {
		uint node = nodes[i];
		uint left = children[(i - leaves_end) * 2 + 0];
		uint right = children[(i - leaves_end) * 2 + 1];

		float4 l = projections[left * num_sites + site];
		float4 r = projections[right * num_sites + site];
		float4 like = l * r;

		uint tbase = i * 4;
		float p0 = dot(transitions[tbase + 0], like);
		float p1 = dot(transitions[tbase + 1], like);
		float p2 = dot(transitions[tbase + 2], like);
		float p3 = dot(transitions[tbase + 3], like);
		float4 projection = float4(p0, p1, p2, p3);

		bool should_scale = projection.x < scale
			&& projection.y < scale
			&& projection.z < scale
			&& projection.w < scale;

		uint proj_idx = node * num_sites + site;
		uchar old_scale = scales[proj_idx];
		if (should_scale) {
			projection *= inv_scale;
			if (old_scale == 0) {
				scale_sum += scale_ln;
				scales[proj_idx] = 1;
			}
		} else if (old_scale != 0) {
			scale_sum -= scale_ln;
			scales[proj_idx] = 0;
		}

		projections[proj_idx] = projection;
	}

	uint root = nodes[num_updated];
	uint left = children[(num_updated - leaves_end) * 2 + 0];
	uint right = children[(num_updated - leaves_end) * 2 + 1];

	float4 left_projection = projections[left * num_sites + site];
	float4 right_projection = projections[right * num_sites + site];
	float4 lk = (left_projection * right_projection) * frequencies;

	float sum = lk.x + lk.y + lk.z + lk.w;
	likelihoods[site] = log(sum);

	uint root_idx = root * num_sites + site;
	if (scales[root_idx] != 0) {
		scales[root_idx] = 0;
		scale_sum -= scale_ln;
	}

	scale_sums[site] = scale_sum;
}
