#include <metal_stdlib>
using namespace metal;

// Build leaf projection vector for one site.
// The leaf is an ambiguity bitmask:
// bit0=A, bit1=C, bit2=G, bit3=T.
// For each allowed nucleotide we add the corresponding transition-column
// contribution into every parent state component.
static inline float4 calc_leaf_projection(uchar leaf, float4 t0, float4 t1, float4 t2, float4 t3) {
	float4 out = float4(0.0f);
	if (leaf & 0b0001) out += float4(t0.x, t1.x, t2.x, t3.x);
	if (leaf & 0b0010) out += float4(t0.y, t1.y, t2.y, t3.y);
	if (leaf & 0b0100) out += float4(t0.z, t1.z, t2.z, t3.z);
	if (leaf & 0b1000) out += float4(t0.w, t1.w, t2.w, t3.w);
	return out;
}

// Main proposal kernel.
//
// Thread mapping:
// - one thread == one site
// - each thread processes all nodes in topological update order for its site
//
// This matches CPU/CUDA algorithmic structure:
// 1) update leaf-originating projections
// 2) propagate through updated internal nodes
// 3) compute root log-likelihood
// 4) maintain scaling bookkeeping to avoid underflow
kernel void propose_kernel(
	const device uchar* leaves [[buffer(0)]],
	device float4* projections [[buffer(1)]],
	device uchar* scales [[buffer(2)]],
	device uint* scale_sums [[buffer(3)]],
	const device uint* nodes [[buffer(4)]],
	const device uint* children [[buffer(5)]],
	const device float4* transitions [[buffer(6)]],
	device float* likelihoods [[buffer(7)]],
	const device uint* params_u32 [[buffer(8)]],
	const device float* params_f32 [[buffer(9)]],
	const device float4* freq_buf [[buffer(10)]],
	uint gid [[thread_position_in_grid]]
) {
	// Packed runtime parameters (written by Rust before dispatch).
	uint num_sites = params_u32[0];
	uint num_updated = params_u32[1];
	uint leaves_end = params_u32[2];
	uint scale_ln = params_u32[3];
	float scale = params_f32[0];
	float inv_scale = params_f32[1];
	float4 frequencies = freq_buf[0];

	// Extra threads outside site range are harmlessly dropped.
	if (gid >= num_sites) {
		return;
	}

	uint site = gid;
	// Start with accepted scale sum for this site; update it incrementally.
	uint scale_sum = scale_sums[site];

	// 1) Leaf prefix of update list.
	// For each leaf edge i:
	// projection(parent_state) = sum_{observed_states at leaf} T[parent_state, observed_state]
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

	// 2) Updated internal nodes (excluding root).
	// For parent node i:
	// like = projection(left_child) * projection(right_child)
	// projection(parent_state) = dot(transition_row[parent_state], like)
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

		// Scaling rule (same as CPU/CUDA):
		// if all 4 components are smaller than `scale`, multiply by `inv_scale`
		// and record +scale_ln in the per-site accumulator.
		bool should_scale = projection.x < scale
			&& projection.y < scale
			&& projection.z < scale
			&& projection.w < scale;

		// `scales[proj_idx]` remembers whether accepted state had this node/site scaled.
		// We adjust `scale_sum` only when the scaling flag actually changes.
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

	// 3) Root likelihood for this site.
	// Root itself has no transition, so combine its two children directly:
	// lk = (left * right) * frequencies
	// log_likelihood_site = log(sum(lk))
	uint root = nodes[num_updated];
	uint left = children[(num_updated - leaves_end) * 2 + 0];
	uint right = children[(num_updated - leaves_end) * 2 + 1];

	float4 left_projection = projections[left * num_sites + site];
	float4 right_projection = projections[right * num_sites + site];
	float4 lk = (left_projection * right_projection) * frequencies;

	float sum = lk.x + lk.y + lk.z + lk.w;
	likelihoods[site] = log(sum);

	// Root is treated as unscaled in final likelihood stage.
	// If root scale flag was set, undo its contribution.
	uint root_idx = root * num_sites + site;
	if (scales[root_idx] != 0) {
		scales[root_idx] = 0;
		scale_sum -= scale_ln;
	}

	// Persist updated per-site scaling offset.
	scale_sums[site] = scale_sum;
}
