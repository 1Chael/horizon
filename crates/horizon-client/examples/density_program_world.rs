use horizon_core::{
    density::graph::DensityCompiler,
    density::{DensitySampler, SampleError, SamplerContext},
    noise::DensityVolume,
    position::ChunkPos,
};
use horizon_world::{
    DATA_VERSION,
    heightmap::HeightmapState,
    nbt::Tag,
    registry::{BlockRegistry, BlockStateId},
    worldgen::{
        aquifer::{DisabledAquifer, FluidPicker, FluidStatus},
        density::DensityRegistry,
        router::{NoiseRouter, RouterField},
        terrain::{TerrainBlockStates, TerrainChunk, fill_noise},
    },
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy)]
struct State(u32);
impl HeightmapState for State {
    fn is_air(self) -> bool {
        self.0 < 2
    }
    fn is_air_block(self) -> bool {
        self.0 == 0
    }
    fn has_fluid(self) -> bool {
        self.0 == 3
    }
    fn blocks_motion_in_heightmap(self) -> bool {
        self.0 == 2
    }
    fn blocks_motion_in_heightmap_no_leaves(self) -> bool {
        self.0 == 2
    }
}
struct Properties;
impl TerrainBlockStates for Properties {
    type State = State;
    fn heightmap_state(&self, id: BlockStateId) -> State {
        State(id.0)
    }
}
#[derive(Debug)]
struct Picker;
impl FluidPicker for Picker {
    fn compute(&self, _: [i32; 3]) -> FluidStatus {
        FluidStatus {
            level: -64,
            state: BlockStateId(3),
        }
    }
}

#[derive(Debug)]
struct Recorder {
    inner: Arc<dyn DensitySampler>,
    values: Mutex<Vec<f32>>,
}
impl DensitySampler for Recorder {
    fn sample_value(
        &self,
        context: &mut SamplerContext,
        position: [i32; 3],
    ) -> Result<f32, SampleError> {
        self.inner.sample_value(context, position)
    }
    fn sample_volume(
        &self,
        context: &mut SamplerContext,
        output: &mut [f32],
        volume: DensityVolume,
    ) -> Result<(), SampleError> {
        self.inner.sample_volume(context, output, volume)?;
        self.values.lock().unwrap().copy_from_slice(output);
        Ok(())
    }
}

pub struct Outputs {
    pub fill_noise: Vec<f32>,
    pub direct: Vec<f32>,
    pub solid_blocks: usize,
}

pub fn trivial_final_density() -> Arc<dyn DensitySampler> {
    let mut registry = DensityRegistry::from_json(&BTreeMap::new(), &BTreeMap::new()).unwrap();
    let router = NoiseRouter::from_json(
        r#"{
            "temperature": 0,
            "vegetation": 0,
            "continents": 0,
            "erosion": 0,
            "depth": 0,
            "ridges": 0,
            "chunk_surface_level": 0,
            "final_density": {"type":"minecraft:add", "left":0.125, "right":0.125}
        }"#,
        &mut registry,
    )
    .unwrap();
    let mut compiler = DensityCompiler::new_seeded(registry.into_graph(), 0, false);
    router
        .compile(&mut compiler)
        .unwrap()
        .get(RouterField::FinalDensity)
        .clone()
}

pub fn fill_and_capture(final_density: Arc<dyn DensitySampler>) -> Outputs {
    let registry = BlockRegistry::from_nbt(&Tag::compound([
        ("DataVersion", Tag::Int(DATA_VERSION)),
        (
            "states",
            Tag::List(vec![
                Tag::compound([("id", Tag::String("minecraft:air".into()))]),
                Tag::compound([("id", Tag::String("minecraft:void_air".into()))]),
                Tag::compound([("id", Tag::String("minecraft:stone".into()))]),
                Tag::compound([("id", Tag::String("minecraft:water".into()))]),
            ]),
        ),
        (
            "defaults",
            Tag::compound([
                ("minecraft:air", Tag::Int(0)),
                ("minecraft:void_air", Tag::Int(1)),
                ("minecraft:stone", Tag::Int(2)),
                ("minecraft:water", Tag::Int(3)),
            ]),
        ),
    ]))
    .unwrap();
    let volume = DensityVolume::new([16; 3], [0; 3], [1; 3]).unwrap();
    let mut direct = vec![0.0; volume.size()];
    final_density
        .sample_volume(&mut SamplerContext::sparse(), &mut direct, volume)
        .unwrap();
    let recorder = Arc::new(Recorder {
        inner: final_density,
        values: Mutex::new(vec![0.0; volume.size()]),
    });
    let mut chunk = TerrainChunk::new(ChunkPos::new(0, 0), 0, 16, &registry, Properties).unwrap();
    fill_noise(
        &mut chunk,
        volume,
        recorder.as_ref(),
        &mut SamplerContext::sparse(),
        &mut DisabledAquifer::new(Picker, BlockStateId(0)),
        BlockStateId(2),
    )
    .unwrap();
    let fill_noise = recorder.values.lock().unwrap().clone();
    let solid_blocks = (0..16)
        .flat_map(|y| (0..16).flat_map(move |z| (0..16).map(move |x| (x, y, z))))
        .filter(|&(x, y, z)| chunk.block(x, y, z) == BlockStateId(2))
        .count();
    Outputs {
        fill_noise,
        direct,
        solid_blocks,
    }
}
