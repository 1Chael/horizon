//! Compare a WGSL f32 Perlin kernel with horizon-core's per-sample bits when
//! the f64 coordinate work is precomputed per axis. The separate sqrt diagnostic
//! lives in `wgpu_sqrt_probe`.

use horizon_core::noise::{PerlinNoise, SmearedPerlinNoise, wrap};
use horizon_core::random::LegacyRandom;
use std::time::Instant;
use wgpu::util::DeviceExt;

const N: usize = 60_000;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Point {
    cell: [i32; 4],
    off: [f32; 4],
    fade_y: f32,
    amp: f32,
    _pad: [f32; 2],
}

const GRADIENTS: [[f32; 3]; 16] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
    [1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0],
    [-1.0, 1.0, 0.0],
    [0.0, -1.0, -1.0],
];

const SHADER: &str = r#"
struct Point {
    cell: vec4<i32>,
    off: vec4<f32>,
    fade_y: f32,
    amp: f32,
    pad: vec2<f32>,
}
@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> perm: array<u32>;
@group(0) @binding(3) var<storage, read> grads: array<vec4<f32>>;

fn permute(i: i32) -> i32 { return i32(perm[i & 255]); }
fn gradient(i: i32) -> vec3<f32> {
    let g = grads[permute(i) & 15];
    return vec3<f32>(g.x, g.y, g.z);
}
fn smoothstep(v: f32) -> f32 { return v * v * v * (v * (v * 6.0 - 15.0) + 10.0); }
fn lerp1(delta: f32, start: f32, end: f32) -> f32 { return start + delta * (end - start); }
fn lerp2(a: f32, b: f32, v0: f32, v1: f32, v2: f32, v3: f32) -> f32 {
    return lerp1(b, lerp1(a, v0, v1), lerp1(a, v2, v3));
}
fn lerp3(fx: f32, fy: f32, fz: f32, v: array<f32, 8>) -> f32 {
    return lerp1(fz, lerp2(fx, fy, v[0], v[1], v[2], v[3]),
                 lerp2(fx, fy, v[4], v[5], v[6], v[7]));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&points)) { return; }
    let pt = points[i];
    let cx = pt.cell.x;
    let cy = pt.cell.y;
    let cz = pt.cell.z;
    let p = permute(cx);
    let q = permute(cx + 1);
    let lower = permute(p + cy);
    let upper = permute(p + cy + 1);
    let lower_x = permute(q + cy);
    let upper_x = permute(q + cy + 1);
    var rows: array<i32, 8> = array<i32, 8>(lower, lower_x, upper, upper_x,
                                            lower, lower_x, upper, upper_x);
    var values: array<f32, 8>;
    let dx = pt.off.x;
    let dy = pt.off.y;
    let dz = pt.off.z;
    let fx = smoothstep(dx);
    let fy = smoothstep(pt.fade_y);
    let fz = smoothstep(dz);
    for (var j: i32 = 0; j < 8; j = j + 1) {
        let g = gradient(rows[j] + cz + (j >> 2));
        let px = dx - f32(j & 1);
        let py = dy - f32((j >> 1) & 1);
        let pz = dz - f32((j >> 2) & 1);
        values[j] = g.x * px + g.y * py + g.z * pz;
    }
    out[i] = pt.amp * lerp3(fx, fy, fz, values);
}
"#;

#[path = "perlin_probe_diagnostics.rs"]
mod diagnostics;

fn fudge_y(input_y: f64, relative_y: f64, scale: f64) -> f64 {
    let limited = if input_y >= 0.0 && input_y < relative_y {
        input_y
    } else {
        relative_y
    };
    f64::from((limited / scale + f64::from(1e-7_f32)).floor() as i32) * scale
}

fn axis(coord: f64, offset: f64, fudge: Option<f64>) -> (i32, f32, f32) {
    let shifted = wrap(coord) + offset;
    let cell = shifted.floor() as i32;
    let relative = shifted - f64::from(cell);
    let fade = relative as f32;
    let used = match fudge {
        Some(scale) => (relative - fudge_y(coord, relative, scale)) as f32,
        None => relative as f32,
    };
    (cell, used, fade)
}

fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .expect("adapter");
    let info = adapter.get_info();
    println!(
        "adapter: {} ({:?}, {:?}), fp64={}",
        info.name,
        info.device_type,
        info.backend,
        adapter.features().contains(wgpu::Features::SHADER_F64)
    );
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default())).expect("device");

    let mut random = LegacyRandom::new(0x5eed_1234);
    let perlin = PerlinNoise::new(&mut random);
    let smeared = SmearedPerlinNoise::new(&mut LegacyRandom::new(0x5eed_5678), 0.375);
    let mut perm = [0u32; 256];
    for (i, p) in perlin.table().permutations().iter().enumerate() {
        perm[i] = u32::from(*p);
    }
    let mut grads = [[0f32; 4]; 16];
    for (i, g) in GRADIENTS.iter().enumerate() {
        grads[i] = [g[0], g[1], g[2], 0.0];
    }

    // Deterministic coordinates mixing normal terrain, lattice boundaries,
    // chunk edges, the wrap threshold and the Far Lands.
    let mut coords: Vec<[f64; 3]> = Vec::with_capacity(N);
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for i in 0..N {
        let base = match i % 5 {
            0 => 30_000_000.0,
            1 => 1_000_000.0,
            2 => 4096.0,
            3 => 1.0,
            _ => 16.0,
        };
        let jitter = |r: u64| ((r as f64 / u64::MAX as f64) * 2.0 - 1.0) * base;
        let mut p = [jitter(next()), jitter(next()), jitter(next())];
        if i % 7 == 0 {
            p = p.map(f64::round);
        }
        if i % 11 == 0 {
            p[0] = 16_777_215.0;
        }
        if i % 13 == 0 {
            p[2] = -16_777_216.0;
        }
        if i % 17 == 0 {
            p[1] = -64.0;
        }
        coords.push(p);
    }

    let offsets = perlin.table().offsets();
    let mut perlin_points = Vec::with_capacity(N);
    let mut expected = Vec::with_capacity(N);
    let mut expected_smeared = Vec::with_capacity(N);
    for p in &coords {
        let (cx, ox, _fx) = axis(p[0], offsets[0], None);
        let (cy, oy, fy) = axis(p[1], offsets[1], None);
        let (cz, oz, _fz) = axis(p[2], offsets[2], None);
        perlin_points.push(Point {
            cell: [cx, cy, cz, 0],
            off: [ox, oy, oz, 0.0],
            fade_y: fy,
            amp: 1.0,
            _pad: [0.0; 2],
        });
        expected.push(perlin.get3(p[0], p[1], p[2]));
    }
    let mut points = perlin_points.clone();
    // The smeared table differs, so its own permutation and offsets are used.
    let mut smeared_perm = [0u32; 256];
    for (i, p) in smeared.table().permutations().iter().enumerate() {
        smeared_perm[i] = u32::from(*p);
    }
    for (i, p) in coords.iter().enumerate() {
        let sm = &smeared;
        let off = sm.table().offsets();
        let (cx, ox, _) = axis(p[0], off[0], None);
        let (cy, oy, fy) = axis(p[1], off[1], Some(0.375));
        let (cz, oz, _) = axis(p[2], off[2], None);
        points[i] = Point {
            cell: [cx, cy, cz, 0],
            off: [ox, oy, oz, 0.0],
            fade_y: fy,
            amp: 1.0,
            _pad: [0.0; 2],
        };
        expected_smeared.push(sm.get3(p[0], p[1], p[2]));
    }

    let perm_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("perm"),
        contents: bytemuck::cast_slice(&perm),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let grads_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("grads"),
        contents: bytemuck::cast_slice(&grads),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("perlin"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("perlin"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let points_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("points"),
        contents: bytemuck::cast_slice(&perlin_points),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let out_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: perm_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grads_buffer.as_entire_binding(),
            },
        ],
    });

    let run = |permutation: &[u32; 256]| -> Vec<f32> {
        queue.write_buffer(&perm_buffer, 0, bytemuck::cast_slice(permutation));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((N as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buffer, 0, &readback, 0, (N * 4) as u64);
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = readback.slice(..).get_mapped_range().expect("map");
        let values = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();
        values
    };

    let gpu = run(&perm);
    let mismatches = gpu
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!("Perlin f32 (CPU precomputed axes): {mismatches}/{N} mismatches");
    diagnostics::report_fma(&perlin_points, &perm, &gpu, &expected);

    let gpu_smeared = {
        queue.write_buffer(&points_buffer, 0, bytemuck::cast_slice(&points));
        run(&smeared_perm)
    };
    let mismatches_smeared = gpu_smeared
        .iter()
        .zip(&expected_smeared)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!("Smeared Perlin f32: {mismatches_smeared}/{N} mismatches");

    if mismatches > 0 {
        for (i, (a, b)) in gpu.iter().zip(&expected).enumerate() {
            if a.to_bits() != b.to_bits() {
                println!(
                    "first mismatch {i}: gpu {:08x} {} cpu {:08x} {}",
                    a.to_bits(),
                    a,
                    b.to_bits(),
                    b
                );
                break;
            }
        }
    }

    let start = Instant::now();
    for _ in 0..8 {
        let _ = run(&perm);
    }
    println!(
        "Perlin dispatch+readback: {:.2} ms per 60k points",
        start.elapsed().as_secs_f64() * 1000.0 / 8.0
    );
}
