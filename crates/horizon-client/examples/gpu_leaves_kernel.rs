//! WGSL kernel and host-side table layout for `gpu_leaves_probe`.
//!
//! The CPU narrows every separable noise leaf per axis (5 X, 49 Y, 5 Z entries
//! per chunk per layer) and the GPU expands the cartesian product and sums the
//! per-octave Perlin layers. The corner-value association matches
//! `PerlinNoise::add_volume` (bulk path), not the scalar `get3` path:
//! `(gx*px + gz*pz) + gy*py`.

pub const X_N: usize = 5;
pub const Y_N: usize = 49;
pub const Z_N: usize = 5;
pub const POINTS: usize = X_N * Y_N * Z_N;

/// `Axis` plus the layer's Y-scale must be prepared per axis, per chunk, per
/// layer before the dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Axis {
    pub cell: i32,
    pub off: f32,
    pub pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniforms {
    /// x_n, y_n, z_n, layers.
    pub a: [u32; 4],
    /// chunk_count, then three unused words.
    pub b: [u32; 4],
}

pub const SHADER: &str = r#"
struct Axis {
    cell: i32,
    off: f32,
    pad: vec2<f32>,
}
struct Uniforms {
    a: vec4<u32>,
    b: vec4<u32>,
}
@group(0) @binding(0) var<storage, read> x_axes: array<Axis>;
@group(0) @binding(1) var<storage, read> y_axes: array<Axis>;
@group(0) @binding(2) var<storage, read> z_axes: array<Axis>;
@group(0) @binding(3) var<storage, read> perms: array<u32>;
@group(0) @binding(4) var<storage, read> grads: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> amps: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;
@group(0) @binding(7) var<uniform> u: Uniforms;

fn permute(base: u32, i: i32) -> i32 {
    return i32(perms[base + u32(i & 255)]);
}
fn gradient(perm_base: u32, grad_base: u32, i: i32) -> vec3<f32> {
    let g = grads[grad_base + u32(permute(perm_base, i) & 15)];
    return vec3<f32>(g.x, g.y, g.z);
}
fn smoothstep(v: f32) -> f32 { return v * v * v * (v * (v * 6.0 - 15.0) + 10.0); }
fn lerp1(delta: f32, start: f32, end: f32) -> f32 { return start + delta * (end - start); }
fn lerp2(a: f32, b: f32, v0: f32, v1: f32, v2: f32, v3: f32) -> f32 {
    return lerp1(b, lerp1(a, v0, v1), lerp1(a, v2, v3));
}
fn lerp3(fx: f32, fy: f32, fz: f32,
         v0: f32, v1: f32, v2: f32, v3: f32,
         v4: f32, v5: f32, v6: f32, v7: f32) -> f32 {
    return lerp1(fz, lerp2(fx, fy, v0, v1, v2, v3),
                 lerp2(fx, fy, v4, v5, v6, v7));
}
/// One corner value. `p`/`q` are the two already-permuted X columns, `j` is
/// the corner index 0..7, and the X/Z contribution is summed before Y
/// (`PerlinNoise::add_volume`). No local arrays: naga/RADV miscompiles an
/// `array` carried across storage-buffer loop iterations.
fn corner(base: u32, gbase: u32, p: i32, q: i32, cy: i32, cz: i32, j: i32,
          dx: f32, dy: f32, dz: f32) -> f32 {
    let x = select(p, q, (j & 1) != 0);
    let row = permute(base, x + cy + ((j >> 1) & 1));
    let g = gradient(base, gbase, row + cz + (j >> 2));
    return (g.x * (dx - f32(j & 1)) + g.z * (dz - f32((j >> 2) & 1)))
        + g.y * (dy - f32((j >> 1) & 1));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x;
    let per_chunk = u.a.x * u.a.y * u.a.z;
    if (p >= u.b.x * per_chunk) { return; }
    let c = p / per_chunk;
    let rem = p % per_chunk;
    let iz = rem / (u.a.x * u.a.y);
    let rem2 = rem % (u.a.x * u.a.y);
    let ix = rem2 / u.a.y;
    let iy = rem2 % u.a.y;
    var acc = 0.0;
    for (var l: u32 = 0; l < u.a.w; l = l + 1) {
        let ax = x_axes[(c * u.a.x + ix) * u.a.w + l];
        let ay = y_axes[(c * u.a.y + iy) * u.a.w + l];
        let az = z_axes[(c * u.a.z + iz) * u.a.w + l];
        let perm_base = l * 256u;
        let grad_base = l * 16u;
        let p0 = permute(perm_base, ax.cell);
        let q0 = permute(perm_base, ax.cell + 1);
        let dx = ax.off;
        let dy = ay.off;
        let dz = az.off;
        let fx = smoothstep(dx);
        let fy = smoothstep(dy);
        let fz = smoothstep(dz);
        let c0 = corner(perm_base, grad_base, p0, q0, ay.cell, az.cell, 0, dx, dy, dz);
        let c1 = corner(perm_base, grad_base, p0, q0, ay.cell, az.cell, 1, dx, dy, dz);
        let c2 = corner(perm_base, grad_base, p0, q0, ay.cell, az.cell, 2, dx, dy, dz);
        let c3 = corner(perm_base, grad_base, p0, q0, ay.cell, az.cell, 3, dx, dy, dz);
        let c4 = corner(perm_base, grad_base, p0, q0, ay.cell, az.cell, 4, dx, dy, dz);
        let c5 = corner(perm_base, grad_base, p0, q0, ay.cell, az.cell, 5, dx, dy, dz);
        let c6 = corner(perm_base, grad_base, p0, q0, ay.cell, az.cell, 6, dx, dy, dz);
        let c7 = corner(perm_base, grad_base, p0, q0, ay.cell, az.cell, 7, dx, dy, dz);
        acc = acc + amps[l] * lerp3(fx, fy, fz, c0, c1, c2, c3, c4, c5, c6, c7);
    }
    out[p] = acc;
}
"#;
