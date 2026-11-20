//! Compare a compiled CPU density subgraph (spline over noise-valued leaves,
//! then blend) with the same narrowed-f32 composition in WGSL.

use horizon_core::{
    density::{
        CubicSpline, DensitySampler, Interval, SampleError, SamplerContext, SplinePoint,
        graph::{DensityCompiler, DensityGraph, GraphNode},
    },
    noise::{DensityVolume, PerlinNoise},
    random::LegacyRandom,
};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const N: usize = 60_000;

#[derive(Debug)]
struct Field(Arc<[f32]>);

impl DensitySampler for Field {
    fn sample_value(&self, _: &mut SamplerContext, position: [i32; 3]) -> Result<f32, SampleError> {
        self.0
            .get(position[0] as usize)
            .copied()
            .ok_or(SampleError::Volume(
                horizon_core::noise::DensityError::BufferTooShort,
            ))
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Input {
    values: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Output {
    values: [f32; 4],
}

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
fn spline_segment(x: f32, x0: f32, x1: f32, y0: f32, y1: f32, d0: f32, d1: f32) -> f32 {
    let width = x1 - x0;
    let alpha = (x - x0) / width;
    let left = d0 * width - (y1 - y0);
    let right = -d1 * width + (y1 - y0);
    let linear = y0 + alpha * (y1 - y0);
    let correction = left + alpha * (right - left);
    return linear + (alpha * (1.0 - alpha)) * correction;
}
fn spline_segment_fma(x: f32, x0: f32, x1: f32, y0: f32, y1: f32, d0: f32, d1: f32) -> f32 {
    let width = x1 - x0;
    let alpha = (x - x0) / width;
    let delta = y1 - y0;
    let left = fma(d0, width, -delta);
    let right = fma(-d1, width, delta);
    let linear = fma(alpha, delta, y0);
    let correction = fma(alpha, right - left, left);
    return fma(alpha * (1.0 - alpha), correction, linear);
}
fn blend(a: f32, first: f32, second: f32) -> f32 {
    if (a == 0.0) { return first; }
    if (a == 1.0) { return second; }
    return first + a * (second - first);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input)) { return; }
    let v = input[i];
    let x = v.x;
    var s: f32;
    var sf: f32;
    if (x < -2.0) {
        s = 0.0 + -0.1 * (x - -2.0);
        sf = fma(-0.1, x - -2.0, 0.0);
    } else if (x < 0.0) {
        s = spline_segment(x, -2.0, 0.0, 0.0, 0.5, -0.1, 0.4);
        sf = spline_segment_fma(x, -2.0, 0.0, 0.0, 0.5, -0.1, 0.4);
    } else if (x < 2.0) {
        s = spline_segment(x, 0.0, 2.0, 0.5, 1.0, 0.4, 0.1);
        sf = spline_segment_fma(x, 0.0, 2.0, 0.5, 1.0, 0.4, 0.1);
    } else {
        s = 1.0 + 0.1 * (x - 2.0);
        sf = fma(0.1, x - 2.0, 1.0);
    }
    output[i] = vec4<f32>(s, blend(s, v.y, v.z), sf, fma(sf, v.z - v.y, v.y));
}
"#;

fn compare(label: &str, expected: &[f32], actual: &[f32], input: &[Input]) -> usize {
    let mut count = 0;
    for i in 0..expected.len() {
        if expected[i].to_bits() != actual[i].to_bits() {
            count += 1;
            if count == 1 {
                println!(
                    "  first {label} divergence: sample {i}, input={:?}, expected={:08x} {}, gpu={:08x} {}",
                    input[i].values,
                    expected[i].to_bits(),
                    expected[i],
                    actual[i].to_bits(),
                    actual[i]
                );
                let x = input[i].values[0];
                let manual = if x < -2.0 {
                    0.0 + -0.1 * (x - -2.0)
                } else if x < 0.0 {
                    spline_segment_cpu(x, -2.0, 0.0, 0.0, 0.5, -0.1, 0.4)
                } else if x < 2.0 {
                    spline_segment_cpu(x, 0.0, 2.0, 0.5, 1.0, 0.4, 0.1)
                } else {
                    1.0 + 0.1 * (x - 2.0)
                };
                println!(
                    "    independent Rust spline expression: {:08x} {}",
                    manual.to_bits(),
                    manual
                );
            }
        }
    }
    println!("{label}: {count}/{} mismatches", expected.len());
    count
}

fn spline_segment_cpu(x: f32, x0: f32, x1: f32, y0: f32, y1: f32, d0: f32, d1: f32) -> f32 {
    let width = x1 - x0;
    let alpha = (x - x0) / width;
    let left = d0 * width - (y1 - y0);
    let right = -d1 * width + (y1 - y0);
    let linear = y0 + alpha * (y1 - y0);
    let correction = left + alpha * (right - left);
    linear + (alpha * (1.0 - alpha)) * correction
}

fn spline_segment_fma_cpu(x: f32, x0: f32, x1: f32, y0: f32, y1: f32, d0: f32, d1: f32) -> f32 {
    let width = x1 - x0;
    let alpha = (x - x0) / width;
    let delta = y1 - y0;
    let left = d0.mul_add(width, -delta);
    let right = (-d1).mul_add(width, delta);
    let linear = alpha.mul_add(delta, y0);
    let correction = alpha.mul_add(right - left, left);
    (alpha * (1.0 - alpha)).mul_add(correction, linear)
}

fn spline_value_fma_cpu(x: f32) -> f32 {
    if x < -2.0 {
        (-0.1f32).mul_add(x - -2.0, 0.0)
    } else if x < 0.0 {
        spline_segment_fma_cpu(x, -2.0, 0.0, 0.0, 0.5, -0.1, 0.4)
    } else if x < 2.0 {
        spline_segment_fma_cpu(x, 0.0, 2.0, 0.5, 1.0, 0.4, 0.1)
    } else {
        0.1f32.mul_add(x - 2.0, 1.0)
    }
}

fn main() {
    let mut rng_a = LegacyRandom::new(0x6772_6170_685f_6131);
    let mut rng_b = LegacyRandom::new(0x6772_6170_685f_6232);
    let mut rng_coord = LegacyRandom::new(0x6772_6170_685f_6333);
    let noise_a = PerlinNoise::new(&mut rng_a);
    let noise_b = PerlinNoise::new(&mut rng_b);
    let noise_coord = PerlinNoise::new(&mut rng_coord);
    let mut fields = [
        Vec::with_capacity(N),
        Vec::with_capacity(N),
        Vec::with_capacity(N),
    ];
    let mut input = Vec::with_capacity(N);
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for i in 0..N {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let x = (state as i32) as f64 * 0.25;
        let y = ((state.rotate_left(21) as i32) as f64) * 0.125;
        let z = ((state.rotate_left(43) as i32) as f64) * 0.25;
        let mut coordinate = noise_coord.get3(x, y, z);
        if i < 8 {
            coordinate = [-3.0, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 3.0][i];
        }
        let a = noise_a.get3(x, y, z);
        let b = noise_b.get3(x, y, z);
        fields[0].push(coordinate);
        fields[1].push(a);
        fields[2].push(b);
        input.push(Input {
            values: [coordinate, a, b, 0.0],
        });
    }

    let mut graph = DensityGraph::new();
    let coordinate = graph
        .bind(
            Arc::new(Field(fields[0].clone().into())),
            Interval::new(-4.0, 4.0).unwrap(),
            1,
        )
        .unwrap();
    let first = graph
        .bind(
            Arc::new(Field(fields[1].clone().into())),
            Interval::new(-2.0, 2.0).unwrap(),
            1,
        )
        .unwrap();
    let second = graph
        .bind(
            Arc::new(Field(fields[2].clone().into())),
            Interval::new(-2.0, 2.0).unwrap(),
            1,
        )
        .unwrap();
    let spline = CubicSpline::multipoint(
        coordinate,
        vec![
            SplinePoint {
                location: -2.0,
                value: CubicSpline::Constant(0.0),
                derivative: -0.1,
            },
            SplinePoint {
                location: 0.0,
                value: CubicSpline::Constant(0.5),
                derivative: 0.4,
            },
            SplinePoint {
                location: 2.0,
                value: CubicSpline::Constant(1.0),
                derivative: 0.1,
            },
        ],
    )
    .unwrap();
    let spline_id = graph.insert(GraphNode::Spline(spline)).unwrap();
    let root = graph
        .insert(GraphNode::Lerp {
            alpha: spline_id,
            first,
            second,
        })
        .unwrap();
    let mut compiler = DensityCompiler::new(graph);
    let spline_sampler = compiler.get_sampler(spline_id).unwrap();
    let root_sampler = compiler.get_sampler(root).unwrap();
    let mut context = SamplerContext::default();
    let volume = DensityVolume::new([N as i32, 1, 1], [0, 0, 0], [1; 3]).unwrap();
    let mut expected_spline = vec![0.0; N];
    let mut expected_blend = vec![0.0; N];
    spline_sampler
        .sample_volume(&mut context, &mut expected_spline, volume)
        .unwrap();
    root_sampler
        .sample_volume(&mut context, &mut expected_blend, volume)
        .unwrap();
    let mut expected_spline_fma = Vec::with_capacity(N);
    let mut expected_blend_fma = Vec::with_capacity(N);
    for sample in &input {
        let sf = spline_value_fma_cpu(sample.values[0]);
        expected_spline_fma.push(sf);
        expected_blend_fma
            .push((sample.values[2] - sample.values[1]).mul_add(sf, sample.values[1]));
    }

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
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default())).expect("device");
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("density graph leaves"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("density graph results"),
        size: (N * std::mem::size_of::<Output>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("density graph readback"),
        size: output.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("density graph layout"),
        entries: &[0, 1].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: binding == 0,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }),
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("density graph bind group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("density graph composition"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("density graph pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("density graph composition"),
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
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((N as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = readback.slice(..).get_mapped_range().expect("readback map");
    let actual: &[Output] = bytemuck::cast_slice(&mapped);
    let actual_spline: Vec<f32> = actual.iter().map(|v| v.values[0]).collect();
    let actual_blend: Vec<f32> = actual.iter().map(|v| v.values[1]).collect();
    let actual_spline_fma: Vec<f32> = actual.iter().map(|v| v.values[2]).collect();
    let actual_blend_fma: Vec<f32> = actual.iter().map(|v| v.values[3]).collect();
    let total = compare("compiled spline", &expected_spline, &actual_spline, &input)
        + compare(
            "compiled spline + blend",
            &expected_blend,
            &actual_blend,
            &input,
        );
    let fma_total = compare(
        "explicit-fma spline contract",
        &expected_spline_fma,
        &actual_spline_fma,
        &input,
    ) + compare(
        "explicit-fma spline + blend contract",
        &expected_blend_fma,
        &actual_blend_fma,
        &input,
    );
    println!("strict-expression stage mismatches: {total}");
    println!("explicit-fma contract mismatches: {fma_total}");
    let i = 66;
    println!(
        "sample 66 minimal spline input: x={:08x} {}, strict={:08x} {}, explicit-FMA={:08x} {}, gpu-fma={:08x} {}",
        input[i].values[0].to_bits(),
        input[i].values[0],
        expected_spline[i].to_bits(),
        expected_spline[i],
        expected_spline_fma[i].to_bits(),
        expected_spline_fma[i],
        actual_spline_fma[i].to_bits(),
        actual_spline_fma[i]
    );
    drop(mapped);
    readback.unmap();
    assert_eq!(
        total, 0,
        "WGSL strict-expression density graph comparison failed"
    );
    assert_eq!(
        fma_total, 0,
        "WGSL explicit-FMA density graph composition parity failed"
    );
}
