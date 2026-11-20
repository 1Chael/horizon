//! Enumerate WGPU adapters and report their backend/device class and fp64 support.

fn main() {
    let instance = wgpu::Instance::default();
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    println!("available adapters: {}", adapters.len());
    for (index, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!(
            "adapter {index}: {} ({:?}, {:?}), shader_f64={}",
            info.name,
            info.device_type,
            info.backend,
            adapter.features().contains(wgpu::Features::SHADER_F64),
        );
    }
}
