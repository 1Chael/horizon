//! Phase 38 probe: what the surface (material-rule) stage costs per column
//! versus per block, on real overworld proto chunks.
//!
//! surface_stage_probe <resources> [--seed 0] [--chunks N]
//!
//! The probe rebuilds the surface worker exactly the way `build_surface`
//! does (same rule library, same compiled rule, same density sources, same
//! per-column traversal and counter evolution), but times the public pieces
//! and never writes to the chunk. Biome queries are answered uniformly with
//! "minecraft:plains" (the real resolver is a region lookup; the rule-tree
//! evaluation is real), so the rule-cost figures are a plains-biome estimate.
//! Nothing here is committed into the game and no protected file is edited.

use horizon_core::density::SamplerContext;
use horizon_core::noise::DensityVolume;
use horizon_core::position::{BlockPos, ChunkPos};
use horizon_core::random::{LegacyRandom, RandomSource, XoroshiroRandom};
use horizon_world::heightmap::HeightmapState as _;
use horizon_world::heightmap::HeightmapType;
use horizon_world::nbt::Tag;
use horizon_world::registry::{BlockRegistry, BlockStateId};
use horizon_world::worldgen::material_condition::MaterialRuleLibrary;
use horizon_world::worldgen::material_condition::{
    MaterialConditionRegistry, MaterialConditions, MaterialDensitySources, MaterialNoiseSources,
    MaterialRuleError, MaterialSurfaceQueries,
};
use horizon_world::worldgen::{
    density::DensityRegistry,
    router::{NoiseRouter, RouterField},
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

type Error = Box<dyn std::error::Error + Send + Sync>;

const PLAINS: &str = "minecraft:plains";

/// Uniform-biome queries; everything else keeps the trait's defaults.
#[derive(Debug)]
struct ProbeSurface;

impl MaterialSurfaceQueries for ProbeSurface {
    fn surface_depth(&mut self, _: i32, _: i32) -> i32 {
        3
    }
    fn surface_secondary(&mut self, _: i32, _: i32) -> f64 {
        0.0
    }
    fn biome(&mut self, _: BlockPos) -> Result<Option<Arc<str>>, MaterialRuleError> {
        Ok(Some(Arc::from(PLAINS)))
    }
}

fn registry(resources: &Path) -> Result<&'static BlockRegistry, Error> {
    let facts: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(resources.join("facts.json"))?)?;
    fn tag(v: &serde_json::Value) -> Tag {
        match v {
            serde_json::Value::String(s) => Tag::String(s.as_str().into()),
            serde_json::Value::Number(n) => Tag::Int(n.as_i64().unwrap() as i32),
            serde_json::Value::Array(a) => Tag::List(a.iter().map(tag).collect()),
            serde_json::Value::Object(o) => {
                Tag::Compound(o.iter().map(|(k, v)| (k.as_str().into(), tag(v))).collect())
            }
            _ => panic!("invalid registry metadata"),
        }
    }
    let registry = BlockRegistry::from_nbt(&Tag::compound([
        ("DataVersion", Tag::Int(horizon_world::DATA_VERSION)),
        (
            "states",
            Tag::List(
                facts["states"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| tag(&v[4]))
                    .collect(),
            ),
        ),
        ("defaults", tag(&facts["defaults"])),
    ]))?;
    Ok(Box::leak(Box::new(registry)))
}

fn integer_properties(registry: &BlockRegistry) -> BTreeSet<(String, String)> {
    let mut out = BTreeSet::new();
    for i in 0..registry.len() {
        let state = registry.state(BlockStateId(i as u32)).unwrap();
        for (key, value) in &state.properties {
            let integer = !value.is_empty()
                && value
                    .trim_start_matches('-')
                    .bytes()
                    .all(|b| b.is_ascii_digit());
            if integer {
                out.insert((state.name.clone(), key.clone()));
            }
        }
    }
    out
}

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let resources = PathBuf::from(args.next().expect("usage: surface_stage_probe <resources>"));
    let (mut seed, mut chunks) = (0i64, 4usize);
    while let Some(arg) = args.next() {
        let mut next = || args.next().expect("argument value");
        match arg.as_str() {
            "--seed" => seed = next().parse()?,
            "--chunks" => chunks = next().parse()?,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let read_bundle = |name: &str| -> Result<BTreeMap<String, String>, Error> {
        Ok(serde_json::from_str(&std::fs::read_to_string(
            resources.join("worldgen").join(format!("{name}.json")),
        )?)?)
    };
    let registry = registry(&resources)?;
    let data = horizon_world::worldgen::generator::GeneratorData::load(
        &resources.join("worldgen"),
        registry,
    )?;
    let generator = horizon_world::worldgen::generator::OverworldGenerator::for_dimension(
        data.clone(),
        seed,
        "minecraft:normal",
        "minecraft:overworld",
    )?;

    let noises = read_bundle("noises")?;
    let densities_json = read_bundle("densities")?;
    let rules = read_bundle("rules")?;
    let biome_set: BTreeSet<String> = read_bundle("biomes")?.into_keys().collect();
    let conditions = MaterialConditionRegistry::from_json_with_biome_tags(
        &read_bundle("conditions")?,
        &biome_set,
        &BTreeMap::new(),
    )?;
    let settings_json: serde_json::Value =
        serde_json::from_str(&read_bundle("settings")?["minecraft:overworld"])?;
    let mut densities = DensityRegistry::from_json(&noises, &densities_json)?;
    let router =
        NoiseRouter::from_json(&settings_json["noise_router"].to_string(), &mut densities)?;
    let codec =
        horizon_world::registry::BlockStateJsonCodec::new(registry, integer_properties(registry))?;
    let mut library = MaterialRuleLibrary::from_json(
        &rules,
        &conditions,
        &codec,
        densities,
        seed,
        settings_json["legacy_random_source"]
            .as_bool()
            .unwrap_or(false),
    )?;
    let compiled_router = library.compile_router(router)?;
    let preliminary = compiled_router.get(RouterField::ChunkSurfaceLevel).clone();
    let rule = library.rule(
        settings_json["material_rule"]
            .as_str()
            .ok_or("no material_rule")?,
    )?;
    let sea_level = settings_json["sea_level"].as_i64().ok_or("no sea_level")? as i32;

    let (mut t_setup, mut t_traverse, mut t_update_y, mut t_tryapply) = (0f64, 0f64, 0f64, 0f64);
    let (mut cells, mut stone_cells, mut rule_calls) = (0u64, 0u64, 0u64);
    for chunk_index in 0..chunks as i32 {
        let pos = ChunkPos::new(chunk_index, 0);
        let proto = generator.worker().generate_proto(pos)?;
        let mut chunk = proto.terrain;
        if chunk.heightmap(HeightmapType::WorldSurfaceWg).is_none() {
            chunk.prime_heightmaps(&[HeightmapType::WorldSurfaceWg])?;
        }
        let min = chunk.min_y();
        let height = chunk.height();
        let highest = chunk
            .highest_filled_section_index()
            .map(|i| min + i as i32 * 16 + 15)
            .unwrap_or(min - 1);
        let span = highest.wrapping_sub(min).wrapping_add(1).max(1);
        let volume = DensityVolume::new(
            [16, span, 16],
            [pos.x.wrapping_shl(4), min, pos.z.wrapping_shl(4)],
            [1; 3],
        )?;
        let noise_sources = MaterialNoiseSources::from_json(
            LegacyRandom::new(seed).fork_positional(),
            ProbeSurface,
            &noises,
        )?;
        let sources = MaterialDensitySources::new(
            noise_sources,
            volume,
            preliminary.clone(),
            SamplerContext::default(),
        )?;
        let generation =
            horizon_world::worldgen::anchor::WorldGenerationContext::new(min, height, sea_level);
        let root = XoroshiroRandom::new(seed).fork_positional();
        let mut worker: MaterialConditions<XoroshiroRandom, _> =
            MaterialConditions::new(generation, root, sources);
        let rule = worker.add_rule(rule.clone())?;
        let variants = worker.surface_variants(rule)?;
        let hm = chunk
            .heightmap(HeightmapType::WorldSurfaceWg)
            .expect("primed");
        for lx in 0..16 {
            for lz in 0..16 {
                let x = pos.x.wrapping_shl(4) + lx;
                let z = pos.z.wrapping_shl(4) + lz;
                // Per-column setup: top-of-column scan and the x/z gradients.
                let setup = Instant::now();
                let top = hm.first_available(lx, lz)?;
                let gx = hm
                    .first_available((lx + 1).min(15), lz)?
                    .wrapping_sub(hm.first_available((lx - 1).max(0), lz)?);
                let gz = hm
                    .first_available(lx, (lz + 1).min(15))?
                    .wrapping_sub(hm.first_available(lx, (lz - 1).max(0))?);
                let _ = worker
                    .biome_at(BlockPos::new(x, top, z))?
                    .ok_or("missing biome")?;
                worker.update_xz(x, z, gx, gz);
                t_setup += setup.elapsed().as_secs_f64() * 1000.0;

                let mut above = 0i32;
                let mut water = i32::MIN;
                let mut bottom = i32::MAX;
                let mut y = top;
                while y >= min {
                    let cell = Instant::now();
                    let state = chunk.block(x, y, z);
                    let properties = chunk.block_properties(state);
                    if properties.is_air() {
                        above = 0;
                        water = i32::MIN;
                        t_traverse += cell.elapsed().as_secs_f64() * 1000.0;
                    } else if properties.has_fluid() {
                        if water == i32::MIN {
                            water = y.wrapping_add(1);
                        }
                        t_traverse += cell.elapsed().as_secs_f64() * 1000.0;
                    } else {
                        if bottom >= y {
                            bottom = -32512;
                            let mut below = y.wrapping_sub(1);
                            while below >= min - 1 {
                                let below_state = chunk.block(x, below, z);
                                let below_props = chunk.block_properties(below_state);
                                if below_props.is_air() || below_props.has_fluid() {
                                    bottom = below.wrapping_add(1);
                                    break;
                                }
                                below = below.wrapping_sub(1);
                            }
                        }
                        above = above.wrapping_add(1);
                        stone_cells += 1;
                        t_traverse += cell.elapsed().as_secs_f64() * 1000.0;
                        let update = Instant::now();
                        worker.update_y(above, y.wrapping_sub(bottom).wrapping_add(1), water, y);
                        t_update_y += update.elapsed().as_secs_f64() * 1000.0;
                        let above_sel = variants.surface && y >= worker.minimum_surface_level()?;
                        let rule = variants.rule(y, above_sel);
                        let apply = Instant::now();
                        let _ = worker.try_apply(rule, BlockPos::new(x, y, z))?;
                        t_tryapply += apply.elapsed().as_secs_f64() * 1000.0;
                        rule_calls += 1;
                    }
                    cells += 1;
                    y = y.wrapping_sub(1);
                }
            }
        }
    }
    let n = chunks as f64;
    let total = t_setup + t_traverse + t_update_y + t_tryapply;
    println!(
        "surface probe: {n:.0} chunks; per chunk: setup {:.3} ms, traverse {:.3} ms, \
         update_y {:.3} ms, try_apply {:.3} ms ({:.0} calls, {:.2} us/call); total {:.3} ms",
        t_setup / n,
        t_traverse / n,
        t_update_y / n,
        t_tryapply / n,
        rule_calls as f64 / n,
        t_tryapply * 1000.0 / (rule_calls as f64).max(1.0),
        total / n,
    );
    println!(
        "  cells {:.0}/chunk (stone-ish {:.0}), per-block rule {:.2} us, \
         per-column setup {:.2} us",
        cells as f64 / n,
        stone_cells as f64 / n,
        t_tryapply * 1000.0 / (stone_cells as f64).max(1.0),
        t_setup * 1000.0 / (n * 256.0),
    );
    Ok(())
}
