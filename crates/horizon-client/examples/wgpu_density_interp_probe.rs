//! Compare density-cell trilinear interpolation and density blending in WGSL
//! with the narrowed-f32 Rust operation order used by Horizon's samplers.

use wgpu::util::DeviceExt;

const N: usize = 60_000;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Case {
    corners: [f32; 8],
    axis_alpha: [f32; 4],
    blend: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ResultRow {
    trilinear: f32,
    blended: f32,
    pad: [f32; 2],
}

const SHADER: &str = r#"
struct Case { corners: array<f32, 8>, axis_alpha: vec4<f32>, blend: vec4<f32> }
struct ResultRow { trilinear: f32, blended: f32, pad: vec2<f32> }
@group(0) @binding(0) var<storage, read> cases: array<Case>;
@group(0) @binding(1) var<storage, read_write> results: array<ResultRow>;

fn lerp(a: f32, first: f32, second: f32) -> f32 {
    return first + a * (second - first);
}
fn blend(a: f32, first: f32, second: f32) -> f32 {
    if (a == 0.0) { return first; }
    if (a == 1.0) { return second; }
    return first + a * (second - first);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&cases)) { return; }
    let c = cases[i];
    let ax = c.axis_alpha.x;
    let ay = c.axis_alpha.y;
    let az = c.axis_alpha.z;
    let xy0 = lerp(ay, lerp(ax, c.corners[0], c.corners[2]), lerp(ax, c.corners[1], c.corners[3]));
    let xy1 = lerp(ay, lerp(ax, c.corners[4], c.corners[6]), lerp(ax, c.corners[5], c.corners[7]));
    var r: ResultRow;
    r.trilinear = lerp(az, xy0, xy1);
    r.blended = blend(c.blend.x, c.blend.y, c.blend.z);
    r.pad = vec2<f32>(0.0);
    results[i] = r;
}
"#;

fn lerp(a: f32, first: f32, second: f32) -> f32 {
    first + a * (second - first)
}

fn blend(a: f32, first: f32, second: f32) -> f32 {
    if a == 0.0 {
        first
    } else if a == 1.0 {
        second
    } else {
        first + a * (second - first)
    }
}

fn reference(case: &Case) -> ResultRow {
    let c = case.corners;
    let [ax, ay, az, _] = case.axis_alpha;
    let xy0 = lerp(ay, lerp(ax, c[0], c[2]), lerp(ax, c[1], c[3]));
    let xy1 = lerp(ay, lerp(ax, c[4], c[6]), lerp(ax, c[5], c[7]));
    ResultRow {
        trilinear: lerp(az, xy0, xy1),
        blended: blend(case.blend[0], case.blend[1], case.blend[2]),
        pad: [0.0; 2],
    }
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn sample_value(state: &mut u64) -> f32 {
    let bits = next(state);
    let exponent = 110 + ((bits >> 32) % 36) as u32;
    let mantissa_sign = bits as u32 & 0x807f_ffff;
    f32::from_bits(mantissa_sign | (exponent << 23))
}

fn cases() -> Vec<Case> {
    let mut state = 0x7472_696c_696e_6561u64;
    let blend_alpha = [
        0.0,
        -0.0,
        1.0,
        f32::from_bits(1),
        f32::from_bits(0x3f7f_ffff),
        0.25,
        0.5,
        1.25,
    ];
    (0..N)
        .map(|i| {
            let corners = std::array::from_fn(|_| sample_value(&mut state));
            let axis_alpha = [
                sample_value(&mut state).abs() % 1.0,
                sample_value(&mut state).abs() % 1.0,
                sample_value(&mut state).abs() % 1.0,
                0.0,
            ];
            let alpha = blend_alpha[i % blend_alpha.len()];
            Case {
                corners,
                axis_alpha,
                blend: [
                    alpha,
                    sample_value(&mut state),
                    sample_value(&mut state),
                    0.0,
                ],
            }
        })
        .collect()
}

fn compare(label: &str, cases: &[Case], expected: &[f32], actual: &[f32]) -> usize {
    let mut count = 0;
    for i in 0..expected.len() {
        if expected[i].to_bits() != actual[i].to_bits() {
            count += 1;
            if count == 1 {
                println!(
                    "  first {label} divergence: sample {i}, alpha={:?}, corners={:?}, expected={:08x} {}, gpu={:08x} {}",
                    cases[i].axis_alpha,
                    cases[i].corners,
                    expected[i].to_bits(),
                    expected[i],
                    actual[i].to_bits(),
                    actual[i]
                );
            }
        }
    }
    println!("{label}: {count}/{} mismatches", expected.len());
    count
}

fn main() {
    let input = cases();
    let expected: Vec<ResultRow> = input.iter().map(reference).collect();
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
        label: Some("density interpolation cases"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("density interpolation results"),
        size: (input.len() * std::mem::size_of::<ResultRow>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("density interpolation readback"),
        size: output.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("density interpolation layout"),
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
        label: Some("density interpolation bind group"),
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
        label: Some("density interpolation parity"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("density interpolation pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("density interpolation parity"),
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
        pass.dispatch_workgroups((input.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = readback.slice(..).get_mapped_range().expect("readback map");
    let actual: &[ResultRow] = bytemuck::cast_slice(&mapped);
    let trilinear_expected: Vec<f32> = expected.iter().map(|v| v.trilinear).collect();
    let trilinear_actual: Vec<f32> = actual.iter().map(|v| v.trilinear).collect();
    let blend_expected: Vec<f32> = expected.iter().map(|v| v.blended).collect();
    let blend_actual: Vec<f32> = actual.iter().map(|v| v.blended).collect();
    let total = compare(
        "trilinear corner interpolation",
        &input,
        &trilinear_expected,
        &trilinear_actual,
    ) + compare(
        "density blend factor",
        &input,
        &blend_expected,
        &blend_actual,
    );
    println!("total stage mismatches: {total}");
    drop(mapped);
    readback.unmap();
    assert_eq!(total, 0, "WGSL density interpolation parity failed");
}
