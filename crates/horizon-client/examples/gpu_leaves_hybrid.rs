//! Phase 32 prototype: hybrid density graph — GPU separable leaves, CPU
//! pointwise graph. Read-only: it binds leaf samplers through the additive
//! `DensityGraph::rebind` API and changes no production worldgen path.
//!
//! gpu_leaves_hybrid <worldgen bundle dir> [--settings NAME] [--seed S]
//!     [--chunks N]
//!
//! For each chunk it evaluates overworld `final_density` on the far-LOD coarse
//! lattice (`5 x 49 x 5`, cell `4 x 8 x 4`):
//!   reference = the production CPU sampler;
//!   hybrid    = the same graph with every separable leaf rebound to a sampler
//!               that serves GPU leaf values on the lattice and falls back to
//!               the CPU leaf anywhere else.
//! It reports graph time per chunk, GPU leaf time, the absolute density
//! residual, and the number of `density > 0` flips (the far-LOD solid/air
//! threshold, Phase 29).

#[path = "gpu_leaves_gpu.rs"]
mod gpu;
#[path = "gpu_leaves_kernel.rs"]
mod kernel;
#[path = "gpu_leaves_tables.rs"]
mod tables;

use gpu::Gpu;
use horizon_core::density::{DensitySampler, SampleError, SamplerContext, graph::DensityCompiler};
use horizon_core::noise::{DensityVolume, NormalNoise};
use horizon_world::worldgen::{
    decode_noise,
    density::DensityRegistry,
    router::{NoiseRouter, RouterField},
};
use kernel::{POINTS, X_N, Y_N, Z_N};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tables::{build_layers, cpu_lattice};

type Error = Box<dyn std::error::Error + Send + Sync>;

fn resolve(value: &Value, densities: &BTreeMap<String, Value>, depth: usize) -> Value {
    if depth == 0 {
        return value.clone();
    }
    match value {
        Value::String(name) => densities
            .get(name)
            .map(|definition| resolve(definition, densities, depth - 1))
            .unwrap_or_else(|| value.clone()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(key, child)| (key.clone(), resolve(child, densities, depth - 1)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|child| resolve(child, densities, depth - 1))
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn collect_noise_leaves(value: &Value, leaves: &mut Vec<Value>) {
    match value {
        Value::Object(map) => {
            let kind = map.get("type").and_then(Value::as_str).unwrap_or("");
            let shifted = ["shift_x", "shift_y", "shift_z"]
                .iter()
                .any(|key| map.get(*key).is_some_and(|v| !v.is_null()));
            if kind == "minecraft:noise" && !shifted {
                leaves.push(value.clone());
            }
            for child in map.values() {
                collect_noise_leaves(child, leaves);
            }
        }
        Value::Array(items) => items
            .iter()
            .for_each(|child| collect_noise_leaves(child, leaves)),
        _ => {}
    }
}

/// GPU values for one leaf over a contiguous chunk block, addressed by world
/// block coordinate. Off-lattice queries fall back to the CPU leaf sampler.
#[derive(Debug)]
struct LeafLookup {
    /// GPU chunks form a `(width+1) x (height+1)` grid, one chunk wider and
    /// taller than the evaluated block so the far interpolation plane is
    /// covered by real GPU values instead of a CPU fallback.
    width: usize,
    chunks: usize,
    values: Vec<f32>,
    ignore_y: bool,
}

impl LeafLookup {
    #[inline]
    fn get(&self, [x, y, z]: [i32; 3]) -> Option<f32> {
        let cx = x.div_euclid(16);
        let cz = z.div_euclid(16);
        let chunk = cx as usize + cz as usize * self.width;
        let ix = usize::try_from((x - cx * 16) / 4).ok()?;
        let iz = usize::try_from((z - cz * 16) / 4).ok()?;
        let iy = if self.ignore_y {
            0
        } else {
            usize::try_from((y + 64) / 8).ok()?
        };
        if chunk >= self.chunks || ix >= X_N || iy >= Y_N || iz >= Z_N {
            return None;
        }
        Some(self.values[chunk * POINTS + (iz * X_N + ix) * Y_N + iy])
    }
}

#[derive(Debug)]
struct GpuLeaf {
    lookup: LeafLookup,
    fallback: Arc<dyn DensitySampler>,
}

impl DensitySampler for GpuLeaf {
    fn sample_value(
        &self,
        context: &mut SamplerContext,
        position: [i32; 3],
    ) -> Result<f32, SampleError> {
        match self.lookup.get(position) {
            Some(value) => Ok(value),
            None => self.fallback.sample_value(context, position),
        }
    }

    /// Bulk fill: hoist the chunk and X/Z index arithmetic out of the Y loop,
    /// mirroring the CPU leaf's per-column structure. The per-point default
    /// would divide for every Y and lose the leaf's amortisation.
    fn sample_volume(
        &self,
        context: &mut SamplerContext,
        buffer: &mut [f32],
        volume: DensityVolume,
    ) -> Result<(), SampleError> {
        if buffer.len() < volume.size() {
            return Err(horizon_core::noise::DensityError::BufferTooShort.into());
        }
        let [sx, sy, sz] = volume.sizes();
        let mut index = 0;
        for z in 0..sz {
            let bz = volume.block(horizon_core::geometry::Axis::Z, z).unwrap();
            let cz = bz.div_euclid(16);
            let iz = usize::try_from((bz - cz * 16) / 4).unwrap_or(usize::MAX);
            for x in 0..sx {
                let bx = volume.block(horizon_core::geometry::Axis::X, x).unwrap();
                let cx = bx.div_euclid(16);
                let ix = usize::try_from((bx - cx * 16) / 4).unwrap_or(usize::MAX);
                let chunk = cx as usize + cz as usize * self.lookup.width;
                let in_grid = chunk < self.lookup.chunks && ix < X_N && iz < Z_N;
                let base = chunk * POINTS + (iz * X_N + ix) * Y_N;
                for y in 0..sy {
                    let by = volume.block(horizon_core::geometry::Axis::Y, y).unwrap();
                    let value = if in_grid {
                        let iy = if self.lookup.ignore_y {
                            0
                        } else {
                            usize::try_from((by + 64) / 8).unwrap_or(usize::MAX)
                        };
                        if iy < Y_N {
                            self.lookup.values[base + iy]
                        } else {
                            self.fallback.sample_value(context, [bx, by, bz])?
                        }
                    } else {
                        self.fallback.sample_value(context, [bx, by, bz])?
                    };
                    buffer[index] = value;
                    index += 1;
                }
            }
        }
        Ok(())
    }
}

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .ok_or("usage: gpu_leaves_hybrid <worldgen bundle>")?,
    );
    let (mut settings_name, mut seed, mut chunk_count) =
        ("minecraft:overworld".to_owned(), 0i64, 64usize);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--settings" => settings_name = args.next().ok_or("--settings NAME")?,
            "--seed" => seed = args.next().ok_or("--seed S")?.parse()?,
            "--chunks" => chunk_count = args.next().ok_or("--chunks N")?.parse::<usize>()?.max(1),
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let read = |name: &str| -> Result<BTreeMap<String, String>, Error> {
        Ok(serde_json::from_str(&std::fs::read_to_string(
            dir.join(name),
        )?)?)
    };
    let noises = read("noises.json")?;
    let densities = read("densities.json")?;
    let settings = read("settings.json")?;
    let definition = |name: &str| -> Result<NormalNoise, Error> {
        Ok(decode_noise(noises.get(name).ok_or("unknown noise")?)?)
    };
    let mut registry = DensityRegistry::from_json(&noises, &densities)?;
    let setting: Value = serde_json::from_str(&settings[&settings_name])?;
    let router = NoiseRouter::from_json(&setting["noise_router"].to_string(), &mut registry)?;
    let final_density = router.function(RouterField::FinalDensity);
    let densities_json: BTreeMap<String, Value> = densities
        .iter()
        .map(|(name, text)| Ok((name.clone(), serde_json::from_str(text)?)))
        .collect::<Result<_, serde_json::Error>>()?;
    let router_json: Value = serde_json::from_str(&setting["noise_router"].to_string())?;
    let tree = resolve(&router_json["final_density"], &densities_json, 64);
    let mut stubs = Vec::new();
    collect_noise_leaves(&tree, &mut stubs);

    let mut leaves: Vec<(horizon_core::density::graph::FunctionId, Value)> = Vec::new();
    for stub in &stubs {
        let id = registry.decode_functions(&[&stub.to_string()])?[0];
        if leaves.iter().all(|(existing, _)| *existing != id) {
            leaves.push((id, stub.clone()));
        }
    }
    let graph = registry.into_graph();
    let chunks: Vec<(i32, i32)> = (0..chunk_count)
        .map(|i| ((i as i32 % 16) * 16, (i as i32 / 16) * 16))
        .collect();
    let width = chunk_count.min(16);
    let height = chunk_count.div_ceil(16);
    // One extra row and column of chunks: their leading lattice planes are the
    // evaluated block's far boundary planes.
    let gpu_chunks: Vec<(i32, i32)> = (0..=height)
        .flat_map(|z| (0..=width).map(move |x| (x as i32 * 16, z as i32 * 16)))
        .collect();
    println!(
        "{settings_name}: {} GPU leaves, {chunk_count} chunks, seed {seed}",
        leaves.len()
    );

    let gpu = Gpu::new();
    let info = &gpu.adapter_info;
    println!(
        "adapter: {} ({:?}, {:?}), fp64={}",
        info.name,
        info.device_type,
        info.backend,
        gpu.adapter.features().contains(wgpu::Features::SHADER_F64)
    );

    let mut compiler = DensityCompiler::new_seeded(graph.clone(), seed, false);
    let mut reference_compiler = DensityCompiler::new_seeded(graph, seed, false);
    let mut gpu_leaf_ms = 0.0;
    let mut gpu_leaf_bytes = 0u64;
    for (id, stub) in &leaves {
        let name = stub["noise"].as_str().unwrap_or("-");
        let xz_scale = stub["xz_scale"].as_f64().unwrap_or(0.0);
        let y_scale = stub["y_scale"].as_f64().unwrap_or(0.0);
        let layers = build_layers(&definition(name)?, seed, name);
        let started = Instant::now();
        let values = gpu.sample_leaf(&layers, xz_scale, y_scale, &gpu_chunks);
        gpu_leaf_ms += started.elapsed().as_secs_f64() * 1000.0;
        gpu_leaf_bytes += (gpu_chunks.len() * POINTS * 4) as u64;
        let lookup = LeafLookup {
            width: width + 1,
            chunks: gpu_chunks.len(),
            values,
            ignore_y: y_scale == 0.0,
        };
        let fallback: Arc<dyn DensitySampler> = compiler.get_sampler(*id)?;
        let range = compiler.graph().range(*id)?;
        let axes = compiler.graph().domain_axes(*id)?;
        compiler
            .graph_mut()
            .rebind(*id, Arc::new(GpuLeaf { lookup, fallback }), range, axes)?;
    }
    let sampler_reference = reference_compiler.get_sampler(final_density)?;
    let sampler_hybrid = compiler.get_sampler(final_density)?;

    let reference = {
        let started = Instant::now();
        let values = cpu_lattice(&sampler_reference, &chunks, 1);
        (values, started.elapsed().as_secs_f64() * 1000.0)
    };
    let (reference_values, reference_ms) = reference;
    let started = Instant::now();
    let mut hybrid = vec![0f32; chunks.len() * POINTS];
    for (i, &(cx, cz)) in chunks.iter().enumerate() {
        let volume = DensityVolume::new(
            [X_N as i32, Y_N as i32, Z_N as i32],
            [cx, -64, cz],
            [4, 8, 4],
        )?;
        let mut context = SamplerContext::default();
        sampler_hybrid.sample_volume(
            &mut context,
            &mut hybrid[i * POINTS..(i + 1) * POINTS],
            volume,
        )?;
    }
    let hybrid_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut worst = 0f32;
    let mut flips = 0usize;
    for (a, b) in hybrid.iter().zip(&reference_values) {
        worst = worst.max((a - b).abs());
        if (*a > 0.0) != (*b > 0.0) {
            flips += 1;
        }
    }
    let chunks_f = chunk_count as f64;
    println!(
        "reference {:.2} ms; hybrid graph {:.2} ms + GPU leaves {:.2} ms; \
         {:.2}->{:.2} ms/chunk, speedup {:.2}x",
        reference_ms,
        hybrid_ms,
        gpu_leaf_ms,
        reference_ms / chunks_f,
        (hybrid_ms + gpu_leaf_ms) / chunks_f,
        reference_ms / (hybrid_ms + gpu_leaf_ms)
    );
    println!(
        "readback {:.1} KiB/chunk; max |density diff| {:.3e}; \
         threshold flips {} / {}",
        gpu_leaf_bytes as f64 / 1024.0 / chunks_f,
        worst,
        flips,
        chunks.len() * POINTS
    );
    Ok(())
}
