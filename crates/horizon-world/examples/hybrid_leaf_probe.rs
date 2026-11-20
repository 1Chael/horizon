//! Phase 30 probe: what share of a chunk's `final_density` cost is the seeded
//! noise leaves, and are those leaves separable per axis?
//!
//! hybrid_leaf_probe <worldgen bundle dir> [--settings NAME] [--seed S]
//!     [--chunks N]
//!
//! Read-only: decodes the vanilla router, then times, on the CPU only,
//!   * `final_density.sample_volume` over the real block-resolution chunk
//!     volume (16 x 384 x 16, steps 1),
//!   * each unique noise leaf over the 5 x 49 x 5 coarse lattice that the
//!     interpolated wrapper evaluates (cell 4 x 8),
//!   * the same graph with every noise leaf replaced by a constant.
//! It changes no production file and writes nothing.
use horizon_core::density::{
    SamplerContext,
    graph::{DensityCompiler, FunctionId},
};
use horizon_core::noise::DensityVolume;
use horizon_world::worldgen::{
    density::DensityRegistry,
    router::{NoiseRouter, RouterField},
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

type Error = Box<dyn std::error::Error + Send + Sync>;

const LEAF_TYPES: [&str; 5] = [
    "minecraft:noise",
    "minecraft:shift",
    "minecraft:shift_a",
    "minecraft:shift_b",
    "minecraft:old_blended_noise",
];

/// Replace named density references with their definitions, to a fixed depth.
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

/// Every noise leaf in the tree, without deduplicating; interning does that.
fn collect_leaves(value: &Value, leaves: &mut Vec<Value>) {
    match value {
        Value::Object(map) => {
            if map
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| LEAF_TYPES.contains(&kind))
            {
                leaves.push(value.clone());
            }
            for child in map.values() {
                collect_leaves(child, leaves);
            }
        }
        Value::Array(items) => items.iter().for_each(|child| collect_leaves(child, leaves)),
        _ => {}
    }
}

/// Swap every noise leaf for a constant, keeping the rest of the tree.
fn strip_leaves(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            if map
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| LEAF_TYPES.contains(&kind))
            {
                return serde_json::json!({"type": "minecraft:constant", "value": 0.0});
            }
            Value::Object(
                map.iter()
                    .map(|(key, child)| (key.clone(), strip_leaves(child)))
                    .collect(),
            )
        }
        Value::Array(items) => Value::Array(items.iter().map(strip_leaves).collect()),
        _ => value.clone(),
    }
}

fn timed(mut run: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..2 {
        let start = Instant::now();
        run();
        best = best.min(start.elapsed().as_secs_f64() * 1000.);
    }
    best
}

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .ok_or("usage: hybrid_leaf_probe <worldgen bundle>")?,
    );
    let (mut settings_name, mut seed, mut chunks) = ("minecraft:overworld".to_owned(), 0i64, 16);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--settings" => settings_name = args.next().ok_or("--settings NAME")?,
            "--seed" => seed = args.next().ok_or("--seed S")?.parse()?,
            "--chunks" => chunks = args.next().ok_or("--chunks N")?.parse::<usize>()?.max(1),
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
    let mut registry = DensityRegistry::from_json(&noises, &densities)?;
    let setting: Value = serde_json::from_str(&settings[&settings_name])?;
    let router = NoiseRouter::from_json(&setting["noise_router"].to_string(), &mut registry)?;
    let final_density = router.function(RouterField::FinalDensity);
    let densities_json: BTreeMap<String, Value> = densities
        .iter()
        .map(|(name, text)| Ok((name.clone(), serde_json::from_str(text)?)))
        .collect::<Result<_, serde_json::Error>>()?;
    // The router field is usually a named reference; resolve it to the tree.
    let router_json: Value = serde_json::from_str(&setting["noise_router"].to_string())?;
    let tree = resolve(&router_json["final_density"], &densities_json, 64);
    let mut stubs = Vec::new();
    collect_leaves(&tree, &mut stubs);
    let cells: Vec<(i64, i64)> = {
        let mut found = Vec::new();
        fn walk(value: &Value, found: &mut Vec<(i64, i64)>) {
            match value {
                Value::Object(map) => {
                    if map.get("type").and_then(Value::as_str) == Some("minecraft:interpolated") {
                        let pair = (
                            map["cell_size_xz"].as_i64().unwrap_or(-1),
                            map["cell_size_y"].as_i64().unwrap_or(-1),
                        );
                        if !found.contains(&pair) {
                            found.push(pair);
                        }
                    }
                    map.values().for_each(|child| walk(child, found));
                }
                Value::Array(items) => items.iter().for_each(|child| walk(child, found)),
                _ => {}
            }
        }
        walk(&tree, &mut found);
        found
    };
    // Interning gives the post-decoding identity: equal leaves collapse here.
    let mut leaves: Vec<(FunctionId, Value)> = Vec::new();
    for stub in &stubs {
        let text = stub.to_string();
        let id = registry.decode_functions(&[&text])?[0];
        if leaves.iter().all(|(existing, _)| *existing != id) {
            leaves.push((id, stub.clone()));
        }
    }
    let stripped: Value = strip_leaves(&tree);
    let stripped_id = registry.decode_functions(&[&stripped.to_string()])?[0];
    let mut compiler = DensityCompiler::new_seeded(registry.into_graph(), seed, false);
    println!(
        "{settings_name}: {} unique noise leaves, interpolated cells {cells:?}",
        leaves.len()
    );
    // Timing volumes: the real chunk (block resolution) and the coarse lattice.
    let chunk_volume = DensityVolume::new([16, 384, 16], [0, -64, 0], [1; 3])?;
    let lattice = DensityVolume::new([5, 49, 5], [0, -64, 0], [4, 8, 4])?;
    let mut buffer = vec![0f32; chunk_volume.size()];
    let mut lattice_buffer = vec![0f32; lattice.size()];
    let chunk_sampler = compiler.get_sampler(final_density)?;
    let coarse_total = timed(|| {
        let mut context = SamplerContext::default();
        for index in 0..chunks {
            let (cx, cz) = ((index as i32 % 4) * 16, (index as i32 / 4) * 16);
            let volume = DensityVolume::new([5, 49, 5], [cx, -64, cz], [4, 8, 4]).expect("lattice");
            chunk_sampler
                .sample_volume(&mut context, &mut lattice_buffer, volume)
                .expect("density");
        }
    }) / chunks as f64;
    let total = timed(|| {
        let mut context = SamplerContext::default();
        for index in 0..chunks {
            let (cx, cz) = ((index as i32 % 4) * 16, (index as i32 / 4) * 16);
            let volume =
                DensityVolume::new([16, 384, 16], [cx, -64, cz], [1; 3]).expect("chunk volume");
            chunk_sampler
                .sample_volume(&mut context, &mut buffer, volume)
                .expect("density");
        }
    }) / chunks as f64;
    let mut leaf_ms = Vec::new();
    for (id, stub) in &leaves {
        let sampler = compiler.get_sampler(*id)?;
        let ms = timed(|| {
            let mut context = SamplerContext::default();
            for index in 0..chunks {
                let (cx, cz) = ((index as i32 % 4) * 16, (index as i32 / 4) * 16);
                let volume =
                    DensityVolume::new([5, 49, 5], [cx, -64, cz], [4, 8, 4]).expect("lattice");
                sampler
                    .sample_volume(&mut context, &mut lattice_buffer, volume)
                    .expect("leaf");
            }
        }) / chunks as f64;
        leaf_ms.push((ms, stub.clone()));
    }
    let stripped_sampler = compiler.get_sampler(stripped_id)?;
    let without_leaves = timed(|| {
        let mut context = SamplerContext::default();
        for index in 0..chunks {
            let (cx, cz) = ((index as i32 % 4) * 16, (index as i32 / 4) * 16);
            let volume =
                DensityVolume::new([16, 384, 16], [cx, -64, cz], [1; 3]).expect("chunk volume");
            stripped_sampler
                .sample_volume(&mut context, &mut buffer, volume)
                .expect("density");
        }
    }) / chunks as f64;
    let sum: f64 = leaf_ms.iter().map(|(ms, _)| ms).sum();
    println!(
        "chunk final_density {total:.2} ms; same graph on the coarse lattice {coarse_total:.2} ms; \
         leaves evaluated once per coarse lattice \
         ({n} x {points} = {evals} evals) sum to {sum:.2} ms ({:.0}%); \
         graph with every leaf constant {without_leaves:.2} ms ({:.0}% non-leaf)",
        100. * sum / total,
        100. * without_leaves / total,
        n = leaf_ms.len(),
        points = lattice.size(),
        evals = leaf_ms.len() * lattice.size(),
    );
    println!(
        "{:>8}  {:<26} {:<10} {:>16}  separable",
        "ms", "noise", "kind", "scale xz / y"
    );
    leaf_ms.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (ms, stub) in &leaf_ms {
        let kind = stub["type"]
            .as_str()
            .unwrap_or("?")
            .replace("minecraft:", "");
        let noise = stub["noise"].as_str().unwrap_or("-");
        let separable = ["shift_x", "shift_y", "shift_z"]
            .into_iter()
            .all(|key| stub.get(key).map_or(true, |value| value.is_null()));
        let scale = format!(
            "{:.3} / {:.3}",
            stub["xz_scale"].as_f64().unwrap_or(f64::NAN),
            stub["y_scale"].as_f64().unwrap_or(f64::NAN)
        );
        println!(
            "{ms:8.3}  {noise:<26} {kind:<10} {scale:>16}  {}",
            if separable { "yes" } else { "no (shifted)" }
        );
    }
    Ok(())
}
