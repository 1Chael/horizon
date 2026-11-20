//! Independent WGSL sqrt diagnostic; it is not part of the Perlin parity gate.

use wgpu::util::DeviceExt;

const N: usize = 60_000;
const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&input)) { return; }
    output[i] = sqrt(input[i]);
}
"#;

fn ordered_bits(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
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
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default())).expect("device");
    let mut values = Vec::with_capacity(N + 6);
    let mut state = 0xdead_beef_1234_5678u64;
    for _ in 0..N {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push(f32::from_bits((state >> 32) as u32));
    }
    values.extend([
        0.0,
        -0.0,
        1.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
    ]);

    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("sqrt inputs"),
        contents: bytemuck::cast_slice(&values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sqrt outputs"),
        size: (values.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sqrt readback"),
        size: output.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("sqrt layout"),
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
        label: Some("sqrt bindings"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sqrt"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sqrt pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sqrt"),
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
        pass.dispatch_workgroups((values.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = readback.slice(..).get_mapped_range().expect("readback map");
    let actual: &[f32] = bytemuck::cast_slice(&mapped);
    let mut mismatches = 0;
    let mut finite_mismatches = 0;
    let mut subnormal_inputs = 0;
    let mut subnormal_zero_outputs = 0;
    let mut positive_normal_mismatches = 0;
    let mut negative_inputs = 0;
    let mut negative_finite_outputs = 0;
    let mut max_ulp = 0;
    let mut max_abs = 0.0f32;
    for (input, gpu) in values.iter().zip(actual) {
        let cpu = (*input as f64).sqrt() as f32;
        if cpu.to_bits() == gpu.to_bits() {
            continue;
        }
        mismatches += 1;
        if cpu.is_finite() && gpu.is_finite() {
            finite_mismatches += 1;
            if input.is_subnormal() {
                subnormal_inputs += 1;
                if *gpu == 0.0 {
                    subnormal_zero_outputs += 1;
                }
            } else if *input > 0.0 && input.is_finite() {
                positive_normal_mismatches += 1;
            }
            max_abs = max_abs.max((cpu - *gpu).abs());
            max_ulp = max_ulp.max(ordered_bits(cpu).abs_diff(ordered_bits(*gpu)));
        } else if *input < 0.0 {
            negative_inputs += 1;
            if gpu.is_finite() {
                negative_finite_outputs += 1;
            }
        }
    }
    println!(
        "sqrt (f64 oracle narrowed): {mismatches}/{} mismatches",
        values.len()
    );
    println!(
        "  finite mismatches={finite_mismatches} (subnormal inputs={subnormal_inputs}, zero outputs={subnormal_zero_outputs}, positive normal inputs={positive_normal_mismatches}), max finite error={max_ulp} ULP / {max_abs:e} absolute; negative inputs={negative_inputs} (finite GPU outputs={negative_finite_outputs})"
    );
    drop(mapped);
    readback.unmap();
}
