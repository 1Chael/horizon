//! Host-side leaf tables and CPU lattice reference for `gpu_leaves_probe`.
//!
//! `build_layers` mirrors `NormalNoise::create`'s seeding (two positional forks,
//! then two Perlin tables per octave at the vanilla 1.018126... step).
//! `cpu_lattice` evaluates the production CPU leaf sampler on the same
//! 5 x 49 x 5 coarse lattice the GPU expands.

use crate::kernel::{Axis, POINTS, X_N, Y_N, Z_N};
use horizon_core::density::{DensitySampler, SamplerContext};
use horizon_core::noise::{DensityVolume, NormalNoise, PerlinNoise, wrap};
use horizon_core::random::{PositionalRandom, RandomSource, XoroshiroRandom};
use std::sync::Arc;

pub struct Layer {
    pub perm: [u32; 256],
    pub grads: [[f32; 4]; 16],
    pub offsets: [f64; 3],
    pub frequency: f64,
    pub amplitude: f32,
}

impl Layer {
    fn from_perlin(perlin: &PerlinNoise, frequency: f64, amplitude: f32) -> Self {
        let mut perm = [0u32; 256];
        for (i, p) in perlin.table().permutations().iter().enumerate() {
            perm[i] = u32::from(*p);
        }
        let mut grads = [[0f32; 4]; 16];
        for (i, g) in GRADIENTS.iter().enumerate() {
            grads[i] = [g[0], g[1], g[2], 0.0];
        }
        Self {
            perm,
            grads,
            offsets: *perlin.table().offsets(),
            frequency,
            amplitude,
        }
    }
}

const GRADIENTS: [[f32; 3]; 16] = [
    [1., 1., 0.],
    [-1., 1., 0.],
    [1., -1., 0.],
    [-1., -1., 0.],
    [1., 0., 1.],
    [-1., 0., 1.],
    [1., 0., -1.],
    [-1., 0., -1.],
    [0., 1., 1.],
    [0., -1., 1.],
    [0., 1., -1.],
    [0., -1., -1.],
    [1., 1., 0.],
    [0., -1., 1.],
    [-1., 1., 0.],
    [0., -1., -1.],
];

pub fn build_layers(definition: &NormalNoise, seed: i64, name: &str) -> Vec<Layer> {
    let mut base = XoroshiroRandom::new(seed)
        .fork_positional()
        .from_hash_of(name);
    let first = base.fork_positional();
    let second = base.fork_positional();
    let mut layers = Vec::new();
    for octave in definition.octaves() {
        let octave_name = octave.seed();
        let a = PerlinNoise::new(&mut first.from_hash_of(&octave_name));
        let b = PerlinNoise::new(&mut second.from_hash_of(&octave_name));
        let amplitude = (definition.normalization_factor() * octave.amplitude) as f32;
        layers.push(Layer::from_perlin(&a, octave.frequency, amplitude));
        layers.push(Layer::from_perlin(
            &b,
            octave.frequency * 1.018_126_888_217_522_7,
            amplitude,
        ));
    }
    layers
}

pub fn narrow(coord: f64, offset: f64) -> Axis {
    let shifted = wrap(coord) + offset;
    let cell = shifted.floor() as i32;
    Axis {
        cell,
        off: (shifted - f64::from(cell)) as f32,
        pad: [0.0; 2],
    }
}

pub fn cpu_lattice(
    sampler: &Arc<dyn DensitySampler>,
    chunks: &[(i32, i32)],
    threads: usize,
) -> Vec<f32> {
    let mut output = vec![0f32; chunks.len() * POINTS];
    let per = chunks.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (part, target) in output.chunks_mut(per * POINTS).enumerate() {
            let sampler = Arc::clone(sampler);
            let slice = &chunks[part * per..(part * per + target.len() / POINTS).min(chunks.len())];
            scope.spawn(move || {
                let mut context = SamplerContext::default();
                for (i, &(cx, cz)) in slice.iter().enumerate() {
                    let volume = DensityVolume::new(
                        [X_N as i32, Y_N as i32, Z_N as i32],
                        [cx, -64, cz],
                        [4, 8, 4],
                    )
                    .expect("lattice");
                    sampler
                        .sample_volume(
                            &mut context,
                            &mut target[i * POINTS..(i + 1) * POINTS],
                            volume,
                        )
                        .expect("cpu leaf");
                }
            });
        }
    });
    output
}
