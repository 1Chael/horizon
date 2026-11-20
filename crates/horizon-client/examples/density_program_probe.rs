//! Compare the explicit-FMA density-program evaluator with WGSL.

use horizon_core::{
    density::program::{DensityProgram, PerlinProgram, PerlinSample},
    noise::{PerlinNoise, wrap},
    random::LegacyRandom,
};
use wgpu::util::DeviceExt;

#[path = "density_program_world.rs"]
mod world_volume;

const N: usize = 60_000;
const GRADIENTS: [[f32; 4]; 16] = [
    [1., 1., 0., 0.],
    [-1., 1., 0., 0.],
    [1., -1., 0., 0.],
    [-1., -1., 0., 0.],
    [1., 0., 1., 0.],
    [-1., 0., 1., 0.],
    [1., 0., -1., 0.],
    [-1., 0., -1., 0.],
    [0., 1., 1., 0.],
    [0., -1., 1., 0.],
    [0., 1., -1., 0.],
    [0., -1., -1., 0.],
    [1., 1., 0., 0.],
    [0., -1., 1., 0.],
    [-1., 1., 0., 0.],
    [0., -1., -1., 0.],
];

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSample {
    cell: [i32; 4],
    offset: [f32; 4],
    fade_y: f32,
    amp: f32,
    pad: [f32; 2],
}

fn axis(coordinate: f64, offset: f64) -> (i32, f32) {
    let shifted = wrap(coordinate) + offset;
    let cell = shifted.floor() as i32;
    (cell, (shifted - f64::from(cell)) as f32)
}

fn main() {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .expect("adapter");
    let info = adapter.get_info();
    println!(
        "adapter: {} ({:?}, {:?})",
        info.name, info.device_type, info.backend
    );
    let shader_f64 = adapter.features().contains(wgpu::Features::SHADER_F64);
    println!("wgpu 30.0.1; naga 30.0.1; shader_f64={shader_f64}; compilation_options=default");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default())).expect("device");

    let mut random = LegacyRandom::new(0x5eed_1234);
    let noise = PerlinNoise::new(&mut random);
    let program = PerlinProgram::from_permutation(noise.table().permutations());
    let final_density = world_volume::trivial_final_density();
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut inputs = Vec::with_capacity(N);
    let mut gpu_inputs = Vec::with_capacity(N);
    let mut reference = Vec::with_capacity(N);
    let offsets = noise.table().offsets();
    for i in 0..N {
        let scale = [30_000_000.0, 1_000_000.0, 4096.0, 1.0, 16.0][i % 5];
        let jitter = |v: u64| (v as f64 / u64::MAX as f64 * 2.0 - 1.0) * scale;
        let mut point = [jitter(next()), jitter(next()), jitter(next())];
        if i % 7 == 0 {
            point = point.map(f64::round);
        }
        if i % 11 == 0 {
            point[0] = 16_777_215.0;
        }
        if i % 13 == 0 {
            point[2] = -16_777_216.0;
        }
        if i % 17 == 0 {
            point[1] = -64.0;
        }
        let (cx, dx) = axis(point[0], offsets[0]);
        let (cy, dy) = axis(point[1], offsets[1]);
        let (cz, dz) = axis(point[2], offsets[2]);
        let sample = PerlinSample {
            cell: [cx, cy, cz],
            offset: [dx, dy, dz],
            fade_y: dy,
        };
        inputs.push(sample);
        gpu_inputs.push(GpuSample {
            cell: [cx, cy, cz, 0],
            offset: [dx, dy, dz, 0.0],
            fade_y: dy,
            amp: 1.0,
            pad: [0.0; 2],
        });
        reference.push(noise.get3(point[0], point[1], point[2]));
    }
    let ir = DensityProgram::Perlin(program.clone());
    let cpu: Vec<f32> = inputs.iter().map(|sample| ir.evaluate(*sample)).collect();
    let cpu_mismatches = report("CPU IR vs PerlinNoise", &reference, &cpu);

    let permutation: Vec<u32> = noise
        .table()
        .permutations()
        .iter()
        .map(|&v| u32::from(v))
        .collect();
    let points = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("density program inputs"),
        contents: bytemuck::cast_slice(&gpu_inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let perm = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("seeded Perlin permutation"),
        contents: bytemuck::cast_slice(&permutation),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let grads = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Perlin gradients"),
        contents: bytemuck::cast_slice(&GRADIENTS),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("density program output"),
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("density program readback"),
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let leaf_values = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("materialized Perlin leaf values"),
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            points_entry(0, true),
            points_entry(1, false),
            points_entry(2, true),
            points_entry(3, true),
            points_entry(4, true),
        ],
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: points.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: perm.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grads.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: leaf_values.as_entire_binding(),
            },
        ],
    });
    let leaf = || DensityProgram::Perlin(program.clone());
    let cases = [
        ("R1 Perlin", leaf()),
        (
            "R2 lerp interior",
            DensityProgram::Lerp {
                alpha: Box::new(DensityProgram::Constant(0.375)),
                first: Box::new(leaf()),
                second: Box::new(DensityProgram::Constant(-0.25)),
            },
        ),
        (
            "R2 lerp alpha zero",
            DensityProgram::Lerp {
                alpha: Box::new(DensityProgram::Constant(0.0)),
                first: Box::new(leaf()),
                second: Box::new(DensityProgram::Constant(-0.25)),
            },
        ),
        (
            "R2 lerp alpha one",
            DensityProgram::Lerp {
                alpha: Box::new(DensityProgram::Constant(1.0)),
                first: Box::new(leaf()),
                second: Box::new(DensityProgram::Constant(-0.25)),
            },
        ),
        (
            "R3 spline",
            DensityProgram::Spline {
                input: Box::new(leaf()),
                x0: -2.0,
                x1: 2.0,
                y0: -0.75,
                y1: 0.875,
                d0: 0.125,
                d1: -0.25,
            },
        ),
        (
            "R4 coordinate offset",
            DensityProgram::Offset {
                input: Box::new(leaf()),
                delta: [0.375, -0.25, 0.75],
            },
        ),
        (
            "R5 add/multiply/blend composition",
            DensityProgram::Blend {
                alpha: Box::new(DensityProgram::Constant(0.375)),
                first: Box::new(DensityProgram::Add(
                    Box::new(leaf()),
                    Box::new(DensityProgram::Constant(0.125)),
                )),
                second: Box::new(DensityProgram::Multiply(
                    Box::new(leaf()),
                    Box::new(DensityProgram::Constant(-0.5)),
                )),
            },
        ),
        (
            "R6 trivial-router final_density",
            DensityProgram::Add(
                Box::new(DensityProgram::Constant(0.125)),
                Box::new(DensityProgram::Constant(0.125)),
            ),
        ),
    ];
    let mut failed = cpu_mismatches != 0;
    let mut r6_gpu = None;
    for (label, program) in cases {
        let expected: Vec<f32> = inputs
            .iter()
            .map(|sample| program.evaluate_fma(*sample))
            .collect();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(
                if label == "R1 Perlin" || label == "R4 coordinate offset" {
                    program.wgsl_fma()
                } else {
                    program.wgsl_with_precomputed_leaf_fma()
                }
                .into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups((N as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, (N * 4) as u64);
        if label == "R1 Perlin" {
            encoder.copy_buffer_to_buffer(&output, 0, &leaf_values, 0, (N * 4) as u64);
        }
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let mapped = readback.slice(..).get_mapped_range().expect("map");
        let actual: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        readback.unmap();
        failed |= report(&format!("{label}: WGSL vs CPU IR"), &expected, &actual) != 0;
        if label == "R6 trivial-router final_density" {
            r6_gpu = Some(actual);
        }
    }
    let r6_program = DensityProgram::Add(
        Box::new(DensityProgram::Constant(0.125)),
        Box::new(DensityProgram::Constant(0.125)),
    );
    let output = world_volume::fill_and_capture(final_density);
    let expected_ir = vec![r6_program.evaluate(PerlinSample::default()); output.direct.len()];
    let mut phase17_failed = false;
    phase17_failed |= report(
        "Phase 17 (a) fill_noise_inner vs (b) direct final_density volume",
        &output.direct,
        &output.fill_noise,
    ) != 0;
    phase17_failed |= report(
        "Phase 17 (b) production sampler vs IR CPU evaluator",
        &expected_ir,
        &output.direct,
    ) != 0;
    let gpu_volume = &r6_gpu.as_ref().expect("R6 GPU output")[..output.direct.len()];
    phase17_failed |= report(
        "Phase 17 (b) IR CPU evaluator vs (c) IR GPU lowering",
        &expected_ir,
        gpu_volume,
    ) != 0;
    println!(
        "Phase 17 fixed volume: {} cells; default-block voxels={}",
        output.direct.len(),
        output.solid_blocks
    );
    failed |= phase17_failed;
    if failed {
        std::process::exit(1);
    }
}

fn points_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn report(label: &str, expected: &[f32], actual: &[f32]) -> usize {
    let mut mismatches = 0;
    let mut max_abs = 0.0f32;
    let mut max_ulp = 0i64;
    for (i, (a, b)) in expected.iter().zip(actual).enumerate() {
        max_abs = max_abs.max((a - b).abs());
        if a.to_bits() != b.to_bits() {
            mismatches += 1;
            max_ulp = max_ulp.max((ordered(a.to_bits()) - ordered(b.to_bits())).abs());
            if mismatches == 1 {
                println!(
                    "{label} first divergence {i}: expected {:08x}, got {:08x}",
                    a.to_bits(),
                    b.to_bits()
                );
            }
        }
    }
    println!(
        "{label}: {mismatches}/{} mismatches; max_abs={max_abs:e}; max_ulp={max_ulp}",
        expected.len()
    );
    mismatches
}

fn ordered(bits: u32) -> i64 {
    if bits & 0x8000_0000 != 0 {
        i64::from(!bits)
    } else {
        i64::from(bits | 0x8000_0000)
    }
}
