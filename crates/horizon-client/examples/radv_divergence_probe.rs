//! Compare strict and explicit-FMA R1 Perlin contracts on a selected adapter.

use horizon_core::{
    density::program::{PerlinProgram, PerlinSample},
    noise::{PerlinNoise, wrap},
    random::LegacyRandom,
};
use std::{fs, path::PathBuf};
use wgpu::util::DeviceExt;

const N: usize = 60_000;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuPoint {
    cell: [i32; 4],
    off: [f32; 4],
    fade_y: f32,
    amp: f32,
    pad: [f32; 2],
}

fn axis(value: f64, offset: f64) -> (i32, f32) {
    let shifted = wrap(value) + offset;
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
    println!(
        "wgpu 30.0.1; naga 30.0.1; shader_f64={}; pipeline_options=default",
        adapter.features().contains(wgpu::Features::SHADER_F64)
    );
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default())).expect("device");

    let mut random = LegacyRandom::new(0x5eed_1234);
    let noise = PerlinNoise::new(&mut random);
    let program = PerlinProgram::from_permutation(noise.table().permutations());
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut coordinates = Vec::with_capacity(N);
    let mut samples = Vec::with_capacity(N);
    let mut gpu_points = Vec::with_capacity(N);
    for i in 0..N {
        let scale = [30_000_000.0, 1_000_000.0, 4096.0, 1.0, 16.0][i % 5];
        let jitter = |v: u64| (v as f64 / u64::MAX as f64 * 2.0 - 1.0) * scale;
        let mut xyz = [jitter(next()), jitter(next()), jitter(next())];
        if i % 7 == 0 {
            xyz = xyz.map(f64::round);
        }
        if i % 11 == 0 {
            xyz[0] = 16_777_215.0;
        }
        if i % 13 == 0 {
            xyz[2] = -16_777_216.0;
        }
        if i % 17 == 0 {
            xyz[1] = -64.0;
        }
        let offsets = noise.table().offsets();
        let (cx, dx) = axis(xyz[0], offsets[0]);
        let (cy, dy) = axis(xyz[1], offsets[1]);
        let (cz, dz) = axis(xyz[2], offsets[2]);
        coordinates.push(xyz);
        samples.push(PerlinSample {
            cell: [cx, cy, cz],
            offset: [dx, dy, dz],
            fade_y: dy,
        });
        gpu_points.push(GpuPoint {
            cell: [cx, cy, cz, 0],
            off: [dx, dy, dz, 0.0],
            fade_y: dy,
            amp: 1.0,
            pad: [0.0; 2],
        });
    }

    let perm: Vec<u32> = noise
        .table()
        .permutations()
        .iter()
        .map(|&v| u32::from(v))
        .collect();
    let gradients: [[f32; 4]; 16] = [
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
    let points = storage(
        &device,
        "R1 narrowed inputs",
        bytemuck::cast_slice(&gpu_points),
        wgpu::BufferUsages::STORAGE,
    );
    let permutations = storage(
        &device,
        "seeded permutation",
        bytemuck::cast_slice(&perm),
        wgpu::BufferUsages::STORAGE,
    );
    let grads = storage(
        &device,
        "Perlin gradients",
        bytemuck::cast_slice(&gradients),
        wgpu::BufferUsages::STORAGE,
    );
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("R1 output"),
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("R1 readback"),
        size: (N * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let unused_leaf = storage(
        &device,
        "unused leaf binding",
        &vec![0u8; N * 4],
        wgpu::BufferUsages::STORAGE,
    );
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding != 1,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>()
            .as_slice(),
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
                resource: permutations.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grads.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: unused_leaf.as_entire_binding(),
            },
        ],
    });
    let strict_source = PerlinProgram::wgsl().to_owned();
    let mut fma_source = strict_source.clone();
    fma_source = fma_source.replace(
        "output[i] = pt.amp * perlin(pt);",
        "output[i] = pt.amp * perlin_fma(pt);",
    );
    dump_spirv(&strict_source, "strict");
    dump_spirv(&fma_source, "explicit-fma");

    let strict_gpu = run(
        &device,
        &queue,
        &layout,
        &bind,
        &output,
        &readback,
        &strict_source,
    );
    let fma_gpu = run(
        &device,
        &queue,
        &layout,
        &bind,
        &output,
        &readback,
        &fma_source,
    );
    let strict_cpu: Vec<f32> = samples.iter().map(|&s| program.evaluate(s)).collect();
    let fma_cpu: Vec<f32> = samples.iter().map(|&s| program.evaluate_fma(s)).collect();
    let vanilla_cpu: Vec<f32> = coordinates
        .iter()
        .map(|p| noise.get3(p[0], p[1], p[2]))
        .collect();
    report(
        "strict a*b+c CPU vs strict-source WGSL",
        &strict_cpu,
        &strict_gpu,
        &coordinates,
    );
    report(
        "explicit mul_add CPU vs fma() WGSL",
        &fma_cpu,
        &fma_gpu,
        &coordinates,
    );
    println!(
        "CPU strict IR vs PerlinNoise reference: {} mismatches",
        count(&strict_cpu, &vanilla_cpu)
    );

    // Exercise the R1 fade intermediate directly: x^3 is subnormal for x=2^-40.
    let tiny = 2.0f32.powi(-49);
    let tiny_points = vec![
        GpuPoint {
            cell: [0; 4],
            off: [tiny, tiny, tiny, 0.0],
            fade_y: tiny,
            amp: 1.0,
            pad: [0.0; 2],
        };
        N
    ];
    let tiny_buffer = storage(
        &device,
        "subnormal fade inputs",
        bytemuck::cast_slice(&tiny_points),
        wgpu::BufferUsages::STORAGE,
    );
    let tiny_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("subnormal fade bindings"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: tiny_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: permutations.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grads.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: unused_leaf.as_entire_binding(),
            },
        ],
    });
    let fade_source = strict_source.replace(
        "output[i] = pt.amp * perlin(pt);",
        "output[i] = fade(pt.off.x);",
    );
    let tiny_gpu = run(
        &device,
        &queue,
        &layout,
        &tiny_bind,
        &output,
        &readback,
        &fade_source,
    );
    let x2 = tiny * tiny;
    let x3 = x2 * tiny;
    let strict_fade = x3 * (tiny * (tiny * 6.0 - 15.0) + 10.0);
    let ftz_fade = if x3.is_subnormal() { 0.0 } else { strict_fade };
    println!(
        "fade subnormal diagnostic: x={tiny:e} x^3={x3:e} ({:08x}) strict={:08x} FTZ(x^3)={:08x} GPU={:08x} GPU-vs-strict={} GPU-vs-FTZ={}",
        x3.to_bits(),
        strict_fade.to_bits(),
        ftz_fade.to_bits(),
        tiny_gpu[0].to_bits(),
        count(&vec![strict_fade; N], &tiny_gpu),
        count(&vec![ftz_fade; N], &tiny_gpu)
    );
}

fn dump_spirv(source: &str, label: &str) {
    let module = naga::front::wgsl::parse_str(source).expect("parse diagnostic WGSL");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validate diagnostic WGSL");
    let words =
        naga::back::spv::write_vec(&module, &info, &naga::back::spv::Options::default(), None)
            .expect("emit diagnostic SPIR-V");
    let mut path: PathBuf = std::env::temp_dir();
    path.push(format!("horizon-radv-{label}-{}.spv", std::process::id()));
    fs::write(&path, bytemuck::cast_slice(&words)).expect("write diagnostic SPIR-V");
    println!(
        "Naga 30.0.1 SPIR-V ({label}): {} words at {}",
        words.len(),
        path.display()
    );
}

fn storage(
    device: &wgpu::Device,
    label: &str,
    data: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage,
    })
}

fn run(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    bind: &wgpu::BindGroup,
    output: &wgpu::Buffer,
    readback: &wgpu::Buffer,
    source: &str,
) -> Vec<f32> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("R1 contract diagnostic"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups((N as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(output, 0, readback, 0, (N * 4) as u64);
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = readback.slice(..).get_mapped_range().expect("map");
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    readback.unmap();
    values
}

fn ordered(bits: u32) -> i64 {
    if bits & 0x8000_0000 != 0 {
        i64::from(!bits)
    } else {
        i64::from(bits | 0x8000_0000)
    }
}

fn report(label: &str, cpu: &[f32], gpu: &[f32], coords: &[[f64; 3]]) {
    let mut mismatches = 0usize;
    let mut max_ulp = 0i64;
    let mut max_abs = 0.0f32;
    let mut buckets = [0usize; 4]; // 1, 2-16, 17+, exact handled separately
    for (i, (&a, &b)) in cpu.iter().zip(gpu).enumerate() {
        if a.to_bits() == b.to_bits() {
            continue;
        }
        mismatches += 1;
        max_abs = max_abs.max((a - b).abs());
        let ulp = (ordered(a.to_bits()) - ordered(b.to_bits())).abs();
        max_ulp = max_ulp.max(ulp);
        buckets[if ulp == 1 {
            0
        } else if ulp <= 16 {
            1
        } else {
            2
        }] += 1;
        if mismatches <= 10 {
            println!(
                "{label} mismatch {}: index={i} coords={:?} CPU={:08x} GPU={:08x} ULP={ulp}",
                mismatches,
                coords[i],
                a.to_bits(),
                b.to_bits()
            );
        }
    }
    println!(
        "{label}: {mismatches}/{N}; ULP buckets 1={} 2-16={} >16={}; max_ulp={max_ulp}; max_abs={max_abs:e}",
        buckets[0], buckets[1], buckets[2],
    );
}

fn count(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}
