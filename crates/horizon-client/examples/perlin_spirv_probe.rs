//! Perlin density ladder (R1-R6) as hand-written SPIR-V, run through wgpu's
//! SPIR-V passthrough so no naga translation sits between the assembly and
//! the driver.
//!
//! The module is `examples/spv/perlin_ladder.spvasm`: every floating-point
//! add/sub/mul/div carries `NoContraction`, every Rust `f32::mul_add` site is
//! an explicit `GLSL.std.450 Fma`, and no fast-math flags are set. The CPU
//! reference is the same `DensityProgram::evaluate_fma` the WGSL probe
//! compares against, so the two probes' mismatch counts are directly
//! comparable.
//!
//!   cargo run --profile fast -p horizon-client --example perlin_spirv_probe
//!   cargo run --profile fast -p horizon-client --example perlin_spirv_probe -- --float-controls
//!
//! `--float-controls` additionally asks for `DenormPreserve` and
//! `SignedZeroInfNanPreserve` (SPV_KHR_float_controls2). That needs
//! `VK_KHR_shader_float_controls2` on the device; without it shader module
//! creation fails and the probe says so instead of silently dropping the
//! request.
//!
//! Needs `spirv-as` (spirv-tools) on PATH. Tools/perlin_spirv_probe.sh wraps
//! the whole run for a hardware report.

use horizon_core::{
    density::program::{DensityProgram, PerlinProgram, PerlinSample},
    noise::{PerlinNoise, wrap},
    random::LegacyRandom,
};
use std::process::Command;
use wgpu::util::DeviceExt;

const N: usize = 60_000;
const RUNGS: [&str; 8] = [
    "R1 leaf",
    "R2 alpha 0",
    "R2 alpha 0.375",
    "R2 alpha 1",
    "R3 spline",
    "R4 coordinate offset",
    "R5 add/mul/blend",
    "R6 router root",
];
const ASM: &str = include_str!("spv/perlin_ladder.spvasm");
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

/// Adds the SPV_KHR_float_controls2 header to the hand-written module.
fn with_float_controls(asm: &str) -> String {
    let asm = asm.replacen(
        "OpCapability Shader\n",
        "OpCapability Shader\nOpCapability FloatControls2\n \
         OpCapability DenormPreserve\nOpCapability SignedZeroInfNanPreserve\n",
        1,
    );
    let asm = asm.replacen(
        "%g = OpExtInstImport",
        "OpExtension \"SPV_KHR_float_controls2\"\n%g = OpExtInstImport",
        1,
    );
    asm.replacen(
        "OpExecutionMode %main LocalSize 64 1 1\n",
        "OpExecutionMode %main LocalSize 64 1 1\n \
         OpExecutionMode %main DenormPreserve 32\n \
         OpExecutionMode %main SignedZeroInfNanPreserve 32\n",
        1,
    )
}

/// Assembles the module with `spirv-as` and returns the SPIR-V words.
fn assemble(label: &str, text: &str) -> Vec<u32> {
    let dir = std::env::temp_dir().join(format!("perlin-spirv-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let asm = dir.join(format!("{label}.spvasm"));
    let spv = dir.join(format!("{label}.spv"));
    std::fs::write(&asm, text).expect("write asm");
    let status = Command::new("spirv-as")
        .args(["--target-env", "vulkan1.3", "-o"])
        .arg(&spv)
        .arg(&asm)
        .status()
        .expect("spirv-as not found on PATH (install spirv-tools)");
    assert!(status.success(), "spirv-as failed on {label}");
    let bytes = std::fs::read(&spv).expect("read spv");
    assert_eq!(bytes.len() % 4, 0, "{label}: SPIR-V is not word aligned");
    bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// Every float add/sub/mul/div result must be decorated NoContraction; a
/// silently dropped decoration would defeat the whole probe.
fn check_no_contraction(text: &str) -> usize {
    let decorated = text
        .lines()
        .filter(|line| line.starts_with("OpDecorate ") && line.ends_with(" NoContraction"))
        .count();
    let arithmetic = text
        .lines()
        .filter(|line| {
            let line = line.trim_start();
            line.contains(" = OpFAdd ")
                || line.contains(" = OpFSub ")
                || line.contains(" = OpFMul ")
                || line.contains(" = OpFDiv ")
        })
        .count();
    assert_eq!(
        decorated, arithmetic,
        "assembly has {arithmetic} float operations but {decorated} NoContraction decorations"
    );
    decorated
}

const WGSL_CONTROL: &str = r#"
struct Point { cell: vec4<i32>, off: vec4<f32>, fade_y: f32, amp: f32, pad: vec2<f32> }
@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> perm: array<u32>;
@group(0) @binding(3) var<storage, read> grads: array<vec4<f32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output[id.x * 8u] = points[id.x].off.x;
}
"#;

/// Negative control: drop every `NoContraction` decoration, so the driver is
/// free to contract and reassociate exactly as it would for an ordinary
/// shader. If this rung table also reads 0/60,000, the driver was already
/// exact without being told to be, and the contract is not what produced the
/// result.
fn without_no_contraction(asm: &str) -> String {
    asm.lines()
        .filter(|line| !(line.starts_with("OpDecorate ") && line.ends_with(" NoContraction")))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn main() {
    let wgsl_control = std::env::args().any(|a| a == "--wgsl-control");
    let float_controls = std::env::args().any(|a| a == "--float-controls");
    let allow_contraction = std::env::args().any(|a| a == "--allow-contraction");
    let text = if allow_contraction {
        without_no_contraction(ASM)
    } else if float_controls {
        with_float_controls(ASM)
    } else {
        ASM.to_owned()
    };
    let decorated = if allow_contraction {
        0
    } else {
        check_no_contraction(&text)
    };
    println!(
        "assembly: hand-written SPIR-V, {} lines, {decorated} NoContraction decorations, \
         {} explicit Fma",
        text.lines().count(),
        text.matches(" Fma ").count()
    );

    // `Instance::default()` ignores `WGPU_BACKEND`; honour it so the probe can
    // be pinned to Vulkan (SPIR-V passthrough) or to a software ICD by name.
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    descriptor.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::VULKAN);
    let instance = wgpu::Instance::new(descriptor);
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
        &instance, None,
    ))
    .expect("no adapter: this probe needs a Vulkan device (SPIR-V passthrough)");
    let info = adapter.get_info();
    let features = adapter.features();
    println!(
        "adapter: {} ({:?}, {:?}); driver {}; backend {:?}",
        info.name, info.device_type, info.driver, info.driver_info, info.backend
    );
    println!(
        "features: passthrough_shaders={} shader_f64={} float_controls_requested={float_controls}",
        features.contains(wgpu::Features::PASSTHROUGH_SHADERS),
        features.contains(wgpu::Features::SHADER_F64),
    );
    assert_eq!(
        info.backend,
        wgpu::Backend::Vulkan,
        "SPIR-V passthrough only exists on Vulkan; force it with WGPU_BACKEND=vulkan"
    );
    assert!(
        features.contains(wgpu::Features::PASSTHROUGH_SHADERS),
        "adapter does not offer PASSTHROUGH_SHADERS"
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::PASSTHROUGH_SHADERS,
        ..Default::default()
    }))
    .expect("device");
    device.on_uncaptured_error(std::sync::Arc::new(|error| panic!("wgpu error: {error}")));

    let mut random = LegacyRandom::new(0x5eed_1234);
    let noise = PerlinNoise::new(&mut random);
    let program = PerlinProgram::from_permutation(noise.table().permutations());
    let offsets = noise.table().offsets();
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut samples = Vec::with_capacity(N);
    let mut points = Vec::with_capacity(N);
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
        samples.push(PerlinSample {
            cell: [cx, cy, cz],
            offset: [dx, dy, dz],
            fade_y: dy,
        });
        points.push(GpuSample {
            cell: [cx, cy, cz, 0],
            offset: [dx, dy, dz, 0.0],
            fade_y: dy,
            amp: 1.0,
            pad: [0.0; 2],
        });
    }

    let leaf = || DensityProgram::Perlin(program.clone());
    let cases = [
        leaf(),
        DensityProgram::Lerp {
            alpha: Box::new(DensityProgram::Constant(0.0)),
            first: Box::new(leaf()),
            second: Box::new(DensityProgram::Constant(-0.25)),
        },
        DensityProgram::Lerp {
            alpha: Box::new(DensityProgram::Constant(0.375)),
            first: Box::new(leaf()),
            second: Box::new(DensityProgram::Constant(-0.25)),
        },
        DensityProgram::Lerp {
            alpha: Box::new(DensityProgram::Constant(1.0)),
            first: Box::new(leaf()),
            second: Box::new(DensityProgram::Constant(-0.25)),
        },
        DensityProgram::Spline {
            input: Box::new(leaf()),
            x0: -2.0,
            x1: 2.0,
            y0: -0.75,
            y1: 0.875,
            d0: 0.125,
            d1: -0.25,
        },
        DensityProgram::Offset {
            input: Box::new(leaf()),
            delta: [0.375, -0.25, 0.75],
        },
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
        DensityProgram::Add(
            Box::new(DensityProgram::Constant(0.125)),
            Box::new(DensityProgram::Constant(0.125)),
        ),
    ];
    let expected: Vec<Vec<f32>> = cases
        .iter()
        .map(|case| samples.iter().map(|s| case.evaluate_fma(*s)).collect())
        .collect();

    let label = if allow_contraction {
        "free"
    } else if float_controls {
        "fc"
    } else {
        "base"
    };
    let spv = assemble(label, &text);
    println!("spirv: {} words assembled by spirv-as", spv.len());

    let permutation: Vec<u32> = noise
        .table()
        .permutations()
        .iter()
        .map(|&v| u32::from(v))
        .collect();
    // Passthrough: the words go to the driver untouched (no naga validation
    // or rewriting), which is the whole point of this probe.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = if wgsl_control {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wgsl control"),
            source: wgpu::ShaderSource::Wgsl(WGSL_CONTROL.into()),
        })
    } else {
        unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(label),
                entry_points: std::borrow::Cow::Owned(vec![wgpu::PassthroughShaderEntryPoint {
                    name: std::borrow::Cow::Borrowed("main"),
                    workgroup_size: (64, 1, 1),
                }]),
                spirv: Some(std::borrow::Cow::Owned(spv)),
                ..Default::default()
            })
        }
    };
    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("shader module: {error}");
    }
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("perlin spirv ladder"),
        entries: &[
            entry(0, true),
            entry(1, false),
            entry(2, true),
            entry(3, true),
        ],
    });
    let point_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("points"),
        contents: bytemuck::cast_slice(&points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let perm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("permutation"),
        contents: bytemuck::cast_slice(&permutation),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let grad_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("gradients"),
        contents: bytemuck::cast_slice(&GRADIENTS),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_bytes = (N * RUNGS.len() * 4) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ladder output"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ladder readback"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("perlin spirv ladder"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: point_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: perm_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: grad_buf.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("perlin spirv ladder"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("perlin spirv ladder"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("resources: {error}");
    }
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("perlin spirv ladder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((N as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_bytes);
    queue.submit([encoder.finish()]);
    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("submit: {error}");
    }
    read_buf.slice(..).map_async(wgpu::MapMode::Read, |result| {
        if let Err(error) = result {
            eprintln!("map_async failed: {error:?}");
        }
    });
    let polled = device.poll(wgpu::PollType::wait_indefinitely());
    println!("poll: {polled:?}");
    let data = read_buf.slice(..).get_mapped_range().expect("map");
    let gpu: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    read_buf.unmap();

    let mut failed = false;
    println!("rung                     mismatches           max_abs");
    for (rung, label) in RUNGS.iter().enumerate() {
        let actual: Vec<f32> = (0..N).map(|i| gpu[i * RUNGS.len() + rung]).collect();
        let (mismatches, max_abs, first) = compare(&expected[rung], &actual);
        if first.is_some() {
            failed = true;
        }
        println!(
            "{label:<24} {mismatches:>6}/{N}           {max_abs:e}{}",
            first.map_or(String::new(), |(i, want, got)| format!(
                "  first {i}: want {want:08x} got {got:08x}"
            ))
        );
    }
    if failed {
        std::process::exit(1);
    }
}

fn compare(expected: &[f32], actual: &[f32]) -> (usize, f32, Option<(usize, u32, u32)>) {
    let mut first = None;
    let mut mismatches = 0;
    let mut max_abs = 0.0f32;
    for (i, (want, got)) in expected.iter().zip(actual).enumerate() {
        max_abs = max_abs.max((want - got).abs());
        if want.to_bits() != got.to_bits() {
            mismatches += 1;
            first.get_or_insert((i, want.to_bits(), got.to_bits()));
        }
    }
    (mismatches, max_abs, first)
}
