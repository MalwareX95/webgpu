struct Input {
  data: array<u32>;
};

struct Output {
  data: array<u32>;
};

[[group(0), binding(0)]] 
var<storage, read> input: Input;

[[group(0), binding(1)]] 
var<storage, read_write> sum: Output;
var<workgroup> wg_sum: array<u32,16>;

[[stage(compute), workgroup_size(16)]]
fn reduce(
  [[builtin(local_invocation_id)]] local: vec3<u32>,
  [[builtin(global_invocation_id)]] global: vec3<u32>,
  [[builtin(workgroup_id)]] wg_id: vec3<u32>
) {
    wg_sum[local.x] = input.data[global.x];

    for (var offset: u32 = 16u / 2u; offset > 0u;  offset = offset / 2u) {
      workgroupBarrier();

      if (local.x < offset) {
        wg_sum[local.x] = wg_sum[local.x] + wg_sum[local.x + offset];
      }
    }

    if (local.x == 0u) {
      sum.data[wg_id.x] = wg_sum[local.x];
    }
}