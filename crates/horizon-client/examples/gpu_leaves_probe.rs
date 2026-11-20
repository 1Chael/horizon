//! Phase 31 probe: evaluate the separable noise leaves of overworld
//! `final_density` on the GPU coarse lattice, per leaf, batched over chunks.
//!
//! gpu_leaves_probe <worldgen bundle dir> [--settings NAME] [--seed S]
//!     [--chunks N] [--reps R]
//!
//! The CPU narrows each separable leaf per axis (`5 x 49 x 5` lattice points,
//! layers included), the GPU expands the cartesian product and sums the
//! per-octave Perlin layers, and the result is compared with the CPU leaf
//! sampler on the same lattice. Read-only: no production file is changed.

#[path = "gpu_leaves_gpu.rs"]
mod gpu;
#[path = "gpu_leaves_kernel.rs"]
mod kernel;
#[path = "gpu_leaves_tables.rs"]
mod tables;

use gpu::Gpu;
use horizon_core::density::graph::{DensityCompiler, FunctionId};
use horizon_core::noise::NormalNoise;
use horizon_world::worldgen::{
    decode_noise,
    density::DensityRegistry,
    router::{NoiseRouter, RouterField},
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
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

#[allow(dead_code)] // the buffers must outlive the bind group
fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .ok_or("usage: gpu_leaves_probe <worldgen bundle>")?,
    );
    let (mut settings_name, mut seed, mut chunk_count, mut reps) =
        ("minecraft:overworld".to_owned(), 0i64, 256usize, 3usize);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--settings" => settings_name = args.next().ok_or("--settings NAME")?,
            "--seed" => seed = args.next().ok_or("--seed S")?.parse()?,
            "--chunks" => chunk_count = args.next().ok_or("--chunks N")?.parse::<usize>()?.max(1),
            "--reps" => reps = args.next().ok_or("--reps R")?.parse::<usize>()?.max(1),
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
    let _final_density = router.function(RouterField::FinalDensity);
    let densities_json: BTreeMap<String, Value> = densities
        .iter()
        .map(|(name, text)| Ok((name.clone(), serde_json::from_str(text)?)))
        .collect::<Result<_, serde_json::Error>>()?;
    let router_json: Value = serde_json::from_str(&setting["noise_router"].to_string())?;
    let tree = resolve(&router_json["final_density"], &densities_json, 64);
    let mut stubs = Vec::new();
    collect_noise_leaves(&tree, &mut stubs);

    let mut leaves: Vec<(FunctionId, Value)> = Vec::new();
    for stub in &stubs {
        let id = registry.decode_functions(&[&stub.to_string()])?[0];
        if leaves.iter().all(|(existing, _)| *existing != id) {
            leaves.push((id, stub.clone()));
        }
    }
    let mut compiler = DensityCompiler::new_seeded(registry.into_graph(), seed, false);
    let chunks: Vec<(i32, i32)> = (0..chunk_count)
        .map(|i| ((i as i32 % 16) * 16, (i as i32 / 16) * 16))
        .collect();
    println!(
        "{settings_name}: {} separable noise leaves, {chunk_count} chunks, seed {seed}",
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

    let mut gpu_ms = 0.0;
    let mut narrow_ms = 0.0;
    let mut cpu_ms = 0.0;
    let mut bytes = 0u64;
    let mut total_mismatches = 0usize;
    let mut worst = 0f32;
    let mut bit_exact = 0usize;
    println!(
        "{:>26} {:>4} {:>8} {:>8} {:>9} {:>9} {:>9} {:>7}",
        "noise", "lyr", "narrow", "gpu ms", "max|diff|", "max|ref|", "rel", "flips"
    );
    for (id, stub) in &leaves {
        let name = stub["noise"].as_str().unwrap_or("-");
        let parsed = definition(name)?;
        let xz_scale = stub["xz_scale"].as_f64().unwrap_or(0.0);
        let y_scale = stub["y_scale"].as_f64().unwrap_or(0.0);
        let layers = build_layers(&parsed, seed, name);
        // The CPU narrows every axis (the `prepare` phase); the GPU expands the
        // cartesian product. Time them separately.
        let started = Instant::now();
        let prepared = gpu.prepare(&layers, xz_scale, y_scale, &chunks);
        let leaf_narrow = started.elapsed().as_secs_f64() * 1000.0;
        narrow_ms += leaf_narrow;
        let sampler = compiler.get_sampler(*id)?;
        let reference = cpu_lattice(&sampler, &chunks, 1);
        let mut best = f64::INFINITY;
        let mut gpu_values = Vec::new();
        for _ in 0..reps {
            let started = Instant::now();
            gpu_values = gpu.run(&prepared);
            best = best.min(started.elapsed().as_secs_f64() * 1000.0);
        }
        gpu_ms += best;
        bytes += prepared.bytes;
        let mut mismatches = 0usize;
        let mut leaf_worst = 0f32;
        let mut leaf_scale = 0f32;
        let mut flips = 0usize;
        for (a, b) in gpu_values.iter().zip(&reference) {
            if a.to_bits() != b.to_bits() {
                mismatches += 1;
            }
            let diff = (a - b).abs();
            leaf_worst = leaf_worst.max(diff);
            leaf_scale = leaf_scale.max(b.abs());
            worst = worst.max(diff);
            if (*a > 0.0) != (*b > 0.0) {
                flips += 1;
            }
        }
        total_mismatches += mismatches;
        if mismatches == 0 {
            bit_exact += 1;
        }
        println!(
            "{name:>26} {:>4} {:>8.2} {:>8.3} {:>9.1e} {:>9.1e} {:>9.1e} {:>7}/{}",
            layers.len(),
            leaf_narrow,
            best,
            leaf_worst,
            leaf_scale,
            leaf_worst / leaf_scale.max(f32::MIN_POSITIVE),
            flips,
            gpu_values.len()
        );
    }

    // The CPU leaf baseline, single thread and 4 workers, over the same chunks.
    for (id, _) in &leaves {
        let sampler = compiler.get_sampler(*id)?;
        let started = Instant::now();
        let _ = cpu_lattice(&sampler, &chunks, 1);
        cpu_ms += started.elapsed().as_secs_f64() * 1000.0;
    }
    let started = Instant::now();
    for (id, _) in &leaves {
        let sampler = compiler.get_sampler(*id)?;
        let _ = cpu_lattice(&sampler, &chunks, 4);
    }
    let cpu4_ms = started.elapsed().as_secs_f64() * 1000.0;

    let total_ms = gpu_ms + narrow_ms;
    let chunks_f = chunk_count as f64;
    println!(
        "\nleaves: {:.2} ms GPU + {:.2} ms narrow = {:.2} ms/tile-set; \
         {:.0} chunks/s GPU-only, {:.0} chunks/s GPU+narrow",
        gpu_ms,
        narrow_ms,
        total_ms,
        chunks_f * 1000.0 / gpu_ms,
        chunks_f * 1000.0 / total_ms
    );
    println!(
        "CPU: {:.2} ms 1 worker, {:.2} ms 4 workers ({:.0} chunks/s); \
         gate 2x CPU/4 = {:.0} chunks/s",
        cpu_ms,
        cpu4_ms,
        chunks_f * 1000.0 / cpu4_ms,
        chunks_f * 2000.0 / cpu4_ms
    );
    println!(
        "readback {:.1} KiB/chunk ({:.1} MiB total); {} of {} leaves bit-exact, \
         {} mismatches, worst |diff| {:.3e}",
        bytes as f64 / 1024.0 / chunks_f,
        bytes as f64 / (1024.0 * 1024.0),
        bit_exact,
        leaves.len(),
        total_mismatches,
        worst
    );
    Ok(())
}
