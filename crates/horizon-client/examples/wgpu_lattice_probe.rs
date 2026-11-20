//! Compare WGSL's f32 Perlin coordinate-to-lattice path with a narrowed-f32
//! Rust reference. This does not change or claim to reproduce the f64 worldgen path.

use horizon_core::noise::PerlinNoise;
use horizon_core::random::LegacyRandom;
use wgpu::util::DeviceExt;

const N: usize = 60_000;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Point {
    xyz: [f32; 4],
    offset: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ResultRow {
    wrapped: [f32; 4],
    shifted: [f32; 4],
    cell: [i32; 4],
    frac: [f32; 4],
    x_hash: [i32; 4],
    corners_a: [i32; 4],
    corners_b: [i32; 4],
}

const SHADER: &str = r#"
struct Point { xyz: vec4<f32>, offset: vec4<f32> }
struct ResultRow {
    wrapped: vec4<f32>, shifted: vec4<f32>, cell: vec4<i32>, frac: vec4<f32>,
    x_hash: vec4<i32>, corners_a: vec4<i32>, corners_b: vec4<i32>,
}
@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read_write> results: array<ResultRow>;
@group(0) @binding(2) var<storage, read> perm: array<u32>;

fn wrap_coord(v: f32) -> f32 {
    if (v >= -16777216.0 && v < 16777216.0) { return v; }
    return v - floor(v / 33554432.0 + 0.5) * 33554432.0;
}
fn permute(i: i32) -> i32 { return i32(perm[u32(i & 255)]); }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&points)) { return; }
    let p = points[i];
    let w = vec3<f32>(wrap_coord(p.xyz.x), wrap_coord(p.xyz.y), wrap_coord(p.xyz.z));
    let shifted = w + p.offset.xyz;
    let cell = vec3<i32>(floor(shifted));
    let frac = shifted - vec3<f32>(cell);
    let px = permute(cell.x);
    let qx = permute(cell.x + 1);
    let lower = permute(px + cell.y);
    let upper = permute(px + cell.y + 1);
    let lower_x = permute(qx + cell.y);
    let upper_x = permute(qx + cell.y + 1);
    let rows = array<i32, 8>(lower, lower_x, upper, upper_x,
                             lower, lower_x, upper, upper_x);
    let c0 = vec4<i32>(
        permute(rows[0] + cell.z) & 15,
        permute(rows[1] + cell.z) & 15,
        permute(rows[2] + cell.z) & 15,
        permute(rows[3] + cell.z) & 15);
    let c1 = vec4<i32>(
        permute(rows[4] + cell.z + 1) & 15,
        permute(rows[5] + cell.z + 1) & 15,
        permute(rows[6] + cell.z + 1) & 15,
        permute(rows[7] + cell.z + 1) & 15);
    var result: ResultRow;
    result.wrapped = vec4<f32>(w, 0.0);
    result.shifted = vec4<f32>(shifted, 0.0);
    result.cell = vec4<i32>(cell, 0);
    result.frac = vec4<f32>(frac, 0.0);
    result.x_hash = vec4<i32>(px, qx, lower, upper);
    result.corners_a = c0;
    result.corners_b = c1;
    results[i] = result;
}
"#;

fn wrap_f32(v: f32) -> f32 {
    if v >= -16_777_216.0 && v < 16_777_216.0 {
        v
    } else {
        v - (v / 33_554_432.0 + 0.5).floor() * 33_554_432.0
    }
}

fn permute(perm: &[u32; 256], index: i32) -> i32 {
    perm[(index & 255) as usize] as i32
}

fn reference(point: &Point, perm: &[u32; 256]) -> ResultRow {
    let wrapped = [
        wrap_f32(point.xyz[0]),
        wrap_f32(point.xyz[1]),
        wrap_f32(point.xyz[2]),
    ];
    let shifted = std::array::from_fn(|i| wrapped[i] + point.offset[i]);
    let cell: [i32; 3] = shifted.map(|v| v.floor() as i32);
    let frac: [f32; 3] = std::array::from_fn(|i| shifted[i] - cell[i] as f32);
    let px = permute(perm, cell[0]);
    let qx = permute(perm, cell[0].wrapping_add(1));
    let lower = permute(perm, px.wrapping_add(cell[1]));
    let upper = permute(perm, px.wrapping_add(cell[1]).wrapping_add(1));
    let lower_x = permute(perm, qx.wrapping_add(cell[1]));
    let upper_x = permute(perm, qx.wrapping_add(cell[1]).wrapping_add(1));
    let rows = [
        lower, lower_x, upper, upper_x, lower, lower_x, upper, upper_x,
    ];
    let corners: [i32; 8] = std::array::from_fn(|i| {
        permute(
            perm,
            rows[i].wrapping_add(cell[2]).wrapping_add((i / 4) as i32),
        ) & 15
    });
    ResultRow {
        wrapped: [wrapped[0], wrapped[1], wrapped[2], 0.0],
        shifted: [shifted[0], shifted[1], shifted[2], 0.0],
        cell: [cell[0], cell[1], cell[2], 0],
        frac: [frac[0], frac[1], frac[2], 0.0],
        x_hash: [px, qx, lower, upper],
        corners_a: corners[..4].try_into().unwrap(),
        corners_b: corners[4..].try_into().unwrap(),
    }
}

fn coordinates(offset: [f32; 3]) -> Vec<[f32; 3]> {
    let boundaries = [
        -100_000_000.0,
        -30_000_000.0,
        -16_777_216.0,
        -16_777_215.0,
        -4096.0,
        -1.000_000_1,
        -1.0,
        -0.999_999_94,
        -0.25,
        0.0,
        0.25,
        0.999_999_94,
        1.0,
        1.000_000_1,
        4096.0,
        16_777_215.0,
        16_777_216.0,
        16_777_218.0,
        30_000_000.0,
        100_000_000.0,
    ];
    let mut out = Vec::with_capacity(N);
    for &boundary in &boundaries {
        out.push([
            boundary - offset[0],
            -boundary - offset[1],
            boundary * 0.5 - offset[2],
        ]);
    }
    let mut state = 0x6a09_e667_f3bc_c909u64;
    while out.len() < N {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let unit = |bits: u64| (bits as u32 as f32) / u32::MAX as f32;
        let axis = |bits: u64| (unit(bits) * 2.0 - 1.0) * 100_000_000.0;
        out.push([
            axis(state),
            axis(state.rotate_left(19)),
            axis(state.rotate_left(41)),
        ]);
    }
    out
}

fn report_stage<T: PartialEq + std::fmt::Debug>(
    name: &str,
    expected: &[T],
    actual: &[T],
    points: &[Point],
) -> usize {
    let mut mismatches = 0;
    for i in 0..expected.len() {
        if expected[i] != actual[i] {
            mismatches += 1;
            if mismatches == 1 {
                println!(
                    "  first {name} divergence: sample {i}, xyz={:?}, expected={:?}, gpu={:?}",
                    &points[i].xyz[..3],
                    expected[i],
                    actual[i]
                );
            }
        }
    }
    println!("{name}: {mismatches}/{} mismatches", expected.len());
    mismatches
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

    let noise = PerlinNoise::new(&mut LegacyRandom::new(0x6c61_7474_6963_65));
    let mut perm = [0u32; 256];
    for (dst, src) in perm.iter_mut().zip(noise.table().permutations()) {
        *dst = u32::from(*src);
    }
    let offset = noise.table().offsets().map(|v| v as f32);
    let points: Vec<Point> = coordinates(offset)
        .into_iter()
        .map(|xyz| Point {
            xyz: [xyz[0], xyz[1], xyz[2], 0.0],
            offset: [offset[0], offset[1], offset[2], 0.0],
        })
        .collect();
    let expected: Vec<ResultRow> = points.iter().map(|p| reference(p, &perm)).collect();

    let point_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("world coordinates"),
        contents: bytemuck::cast_slice(&points),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let perm_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("permutation"),
        contents: bytemuck::cast_slice(&perm),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lattice results"),
        size: (points.len() * std::mem::size_of::<ResultRow>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lattice readback"),
        size: output.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lattice layout"),
        entries: &[0, 1, 2].map(|binding| wgpu::BindGroupLayoutEntry {
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
        }),
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lattice bind group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: point_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: perm_buffer.as_entire_binding(),
            },
        ],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lattice parity"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("lattice parity"),
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("lattice pipeline layout"),
                bind_group_layouts: &[Some(&layout)],
                ..Default::default()
            }),
        ),
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
        pass.dispatch_workgroups((points.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = readback.slice(..).get_mapped_range().expect("map readback");
    let actual: &[ResultRow] = bytemuck::cast_slice(&mapped);

    let mut total = 0;
    for axis in 0..3 {
        total += report_stage(
            &format!("wrap axis {axis}"),
            &expected
                .iter()
                .map(|r| r.wrapped[axis].to_bits())
                .collect::<Vec<_>>(),
            &actual
                .iter()
                .map(|r| r.wrapped[axis].to_bits())
                .collect::<Vec<_>>(),
            &points,
        );
        total += report_stage(
            &format!("shift axis {axis}"),
            &expected
                .iter()
                .map(|r| r.shifted[axis].to_bits())
                .collect::<Vec<_>>(),
            &actual
                .iter()
                .map(|r| r.shifted[axis].to_bits())
                .collect::<Vec<_>>(),
            &points,
        );
        total += report_stage(
            &format!("floor/cell axis {axis}"),
            &expected.iter().map(|r| r.cell[axis]).collect::<Vec<_>>(),
            &actual.iter().map(|r| r.cell[axis]).collect::<Vec<_>>(),
            &points,
        );
        total += report_stage(
            &format!("fract axis {axis}"),
            &expected
                .iter()
                .map(|r| r.frac[axis].to_bits())
                .collect::<Vec<_>>(),
            &actual
                .iter()
                .map(|r| r.frac[axis].to_bits())
                .collect::<Vec<_>>(),
            &points,
        );
    }
    total += report_stage(
        "x-neighbor permutation",
        &expected
            .iter()
            .map(|r| r.x_hash[..2].to_vec())
            .collect::<Vec<_>>(),
        &actual
            .iter()
            .map(|r| r.x_hash[..2].to_vec())
            .collect::<Vec<_>>(),
        &points,
    );
    total += report_stage(
        "corner gradient gather",
        &expected
            .iter()
            .map(|r| [r.corners_a, r.corners_b])
            .collect::<Vec<_>>(),
        &actual
            .iter()
            .map(|r| [r.corners_a, r.corners_b])
            .collect::<Vec<_>>(),
        &points,
    );
    println!("total stage mismatches: {total}");
    drop(mapped);
    readback.unmap();
    assert_eq!(total, 0, "WGSL coordinate-to-lattice parity failed");
}
