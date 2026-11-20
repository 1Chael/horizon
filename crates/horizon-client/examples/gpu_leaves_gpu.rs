//! wgpu dispatch for the separable-leaf kernel, shared by `gpu_leaves_probe`
//! and `gpu_leaves_hybrid`.

use crate::kernel::{POINTS, SHADER, Uniforms, X_N, Y_N, Z_N};
use crate::tables::{Layer, narrow};
use wgpu::util::DeviceExt;

pub struct Gpu {
    #[allow(dead_code)] // keeps the adapter's parent alive for the process
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub layout: wgpu::BindGroupLayout,
    pub pipeline: wgpu::ComputePipeline,
    pub adapter_info: wgpu::AdapterInfo,
}

impl Gpu {
    pub fn new() -> Self {
        // `Instance::default()` ignores `WGPU_BACKEND`; honour it so a backend
        // can be pinned (for example Vulkan + `VK_ICD_FILENAMES`, or `gl`).
        let backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all());
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = backends;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .expect("adapter");
        let adapter_info = adapter.get_info();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&Default::default())).expect("device");
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &(0..8)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 6 {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        } else if binding == 7 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: true }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("leaves"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("leaves"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            instance,
            adapter,
            device,
            queue,
            layout,
            pipeline,
            adapter_info,
        }
    }

    /// Used by `gpu_leaves_hybrid`; `gpu_leaves_probe` times `prepare`/`run`
    /// separately and does not call this.
    #[allow(dead_code)]
    pub fn sample_leaf(
        &self,
        layers: &[Layer],
        xz_scale: f64,
        y_scale: f64,
        chunks: &[(i32, i32)],
    ) -> Vec<f32> {
        let prepared = self.prepare(layers, xz_scale, y_scale, chunks);
        self.run(&prepared)
    }

    pub fn prepare(
        &self,
        layers: &[Layer],
        xz_scale: f64,
        y_scale: f64,
        chunks: &[(i32, i32)],
    ) -> Prepared {
        let n = chunks.len();
        let mut xs = Vec::with_capacity(n * X_N * layers.len());
        let mut ys = Vec::with_capacity(n * Y_N * layers.len());
        let mut zs = Vec::with_capacity(n * Z_N * layers.len());
        let mut perms = Vec::with_capacity(layers.len() * 256);
        let mut grads = Vec::with_capacity(layers.len() * 16);
        let mut amps = Vec::with_capacity(layers.len());
        for layer in layers {
            perms.extend_from_slice(&layer.perm);
            grads.extend_from_slice(&layer.grads);
            amps.push(layer.amplitude);
        }
        // Chunk-major, then axis index, then layer: the kernel's index arithmetic.
        for &(cx, cz) in chunks {
            for ix in 0..X_N {
                for layer in layers {
                    let x = f64::from(cx + ix as i32 * 4) * (xz_scale * layer.frequency);
                    xs.push(narrow(x, layer.offsets[0]));
                }
            }
            for iy in 0..Y_N {
                for layer in layers {
                    let y = f64::from(-64 + iy as i32 * 8) * (y_scale * layer.frequency);
                    ys.push(narrow(y, layer.offsets[1]));
                }
            }
            for iz in 0..Z_N {
                for layer in layers {
                    let z = f64::from(cz + iz as i32 * 4) * (xz_scale * layer.frequency);
                    zs.push(narrow(z, layer.offsets[2]));
                }
            }
        }
        let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let make = |label, data: &[u8]| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: data,
                    usage: storage,
                })
        };
        let x_axes = make("x", bytemuck::cast_slice(&xs));
        let y_axes = make("y", bytemuck::cast_slice(&ys));
        let z_axes = make("z", bytemuck::cast_slice(&zs));
        let perms_buf = make("perms", bytemuck::cast_slice(&perms));
        let grads_buf = make("grads", bytemuck::cast_slice(&grads));
        let amps_buf = make("amps", bytemuck::cast_slice(&amps));
        let uniforms = Uniforms {
            a: [X_N as u32, Y_N as u32, Z_N as u32, layers.len() as u32],
            b: [n as u32, 0, 0, 0],
        };
        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("uniforms"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bytes = (n * POINTS * 4) as u64;
        let out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_axes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: y_axes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: z_axes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: perms_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: grads_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: amps_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });
        Prepared {
            _x_axes: x_axes,
            _y_axes: y_axes,
            _z_axes: z_axes,
            _perms: perms_buf,
            _grads: grads_buf,
            _amps: amps_buf,
            out,
            readback,
            _bind_group: bind_group,
            chunks: n,
            bytes,
        }
    }

    pub fn run(&self, prepared: &Prepared) -> Vec<f32> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &prepared._bind_group, &[]);
            pass.dispatch_workgroups(((prepared.chunks * POINTS) as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&prepared.out, 0, &prepared.readback, 0, prepared.bytes);
        self.queue.submit([encoder.finish()]);
        prepared
            .readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = prepared.readback.slice(..).get_mapped_range().expect("map");
        let values = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        prepared.readback.unmap();
        values
    }
}

/// The buffers must outlive the bind group.
pub struct Prepared {
    _x_axes: wgpu::Buffer,
    _y_axes: wgpu::Buffer,
    _z_axes: wgpu::Buffer,
    _perms: wgpu::Buffer,
    _grads: wgpu::Buffer,
    _amps: wgpu::Buffer,
    out: wgpu::Buffer,
    readback: wgpu::Buffer,
    _bind_group: wgpu::BindGroup,
    pub chunks: usize,
    pub bytes: u64,
}
