//! Small, explicit density programs for CPU/GPU parity experiments.
//!
//! This does not participate in `DensityCompiler` or production worldgen.
//! Inputs to a Perlin leaf are already narrowed to a lattice cell and f32
//! offsets on CPU; this keeps WGSL out of the production f64 coordinate path.

use super::arithmetic::{clamp as clamp_value, java_max, java_min};

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

/// Narrowed input to one seeded Perlin leaf.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PerlinSample {
    pub cell: [i32; 3],
    pub offset: [f32; 3],
    /// The interpolation Y value can differ from `offset[1]` for smeared noise.
    pub fade_y: f32,
}

/// A seeded Perlin program. The permutation is copied from the existing
/// `PerlinNoise` table, which remains responsible for consuming seed draws.
#[derive(Clone, Debug)]
pub struct PerlinProgram {
    permutation: [u8; 256],
}

/// Unary density functions shared by `mode` nodes. `Sqrt` and `Log` are absent:
/// their production definitions are `f64` libm calls whose correctly-rounded
/// `f32` rounding cannot be reproduced by a GPU, so they stay off the tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnaryKind {
    Abs,
    Negate,
    Square,
    Cube,
    HalfNegative,
    QuarterNegative,
    Reciprocal,
    Sign,
    Squeeze,
}

/// Binary density functions shared by `min`, `max`, arithmetic and the wire
/// `add`/`mul` nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryKind {
    Add,
    Subtract,
    Multiply,
    Divide,
    Minimum,
    Maximum,
}

/// A supported pointwise density expression. Variants are added only after
/// their CPU and WGSL lowerings pass the named-adapter parity probe.
#[derive(Clone, Debug)]
pub enum DensityProgram {
    Perlin(PerlinProgram),
    Constant(f32),
    Lerp {
        alpha: Box<Self>,
        first: Box<Self>,
        second: Box<Self>,
    },
    Spline {
        input: Box<Self>,
        x0: f32,
        x1: f32,
        y0: f32,
        y1: f32,
        d0: f32,
        d1: f32,
    },
    Offset {
        input: Box<Self>,
        delta: [f32; 3],
    },
    Add(Box<Self>, Box<Self>),
    Multiply(Box<Self>, Box<Self>),
    Blend {
        alpha: Box<Self>,
        first: Box<Self>,
        second: Box<Self>,
    },
    Clamp {
        input: Box<Self>,
        min: f32,
        max: f32,
    },
    Unary {
        kind: UnaryKind,
        input: Box<Self>,
    },
    Binary {
        kind: BinaryKind,
        left: Box<Self>,
        right: Box<Self>,
    },
}

impl DensityProgram {
    pub fn evaluate(&self, sample: PerlinSample) -> f32 {
        match self {
            Self::Perlin(program) => program.evaluate(sample),
            Self::Constant(value) => *value,
            Self::Lerp {
                alpha,
                first,
                second,
            }
            | Self::Blend {
                alpha,
                first,
                second,
            } => blend(
                alpha.evaluate(sample),
                first.evaluate(sample),
                second.evaluate(sample),
            ),
            Self::Spline {
                input,
                x0,
                x1,
                y0,
                y1,
                d0,
                d1,
            } => spline(input.evaluate(sample), *x0, *x1, *y0, *y1, *d0, *d1),
            Self::Offset { input, delta } => input.evaluate(offset_sample(sample, *delta)),
            Self::Add(left, right) => left.evaluate(sample) + right.evaluate(sample),
            Self::Multiply(left, right) => left.evaluate(sample) * right.evaluate(sample),
            Self::Clamp { input, min, max } => clamp_value(input.evaluate(sample), *min, *max),
            Self::Unary { kind, input } => unary_value(*kind, input.evaluate(sample)),
            Self::Binary { kind, left, right } => {
                binary_value(*kind, left.evaluate(sample), right.evaluate(sample))
            }
        }
    }

    /// Evaluate under the explicit-FMA contract used by hardware parity probes.
    pub fn evaluate_fma(&self, sample: PerlinSample) -> f32 {
        match self {
            Self::Perlin(program) => program.evaluate_fma(sample),
            Self::Constant(value) => *value,
            Self::Lerp {
                alpha,
                first,
                second,
            }
            | Self::Blend {
                alpha,
                first,
                second,
            } => blend_fma(
                alpha.evaluate_fma(sample),
                first.evaluate_fma(sample),
                second.evaluate_fma(sample),
            ),
            Self::Spline {
                input,
                x0,
                x1,
                y0,
                y1,
                d0,
                d1,
            } => spline(input.evaluate_fma(sample), *x0, *x1, *y0, *y1, *d0, *d1),
            Self::Offset { input, delta } => input.evaluate_fma(offset_sample(sample, *delta)),
            Self::Add(left, right) => left.evaluate_fma(sample) + right.evaluate_fma(sample),
            Self::Multiply(left, right) => left.evaluate_fma(sample) * right.evaluate_fma(sample),
            Self::Clamp { input, min, max } => clamp_value(input.evaluate_fma(sample), *min, *max),
            Self::Unary { kind, input } => unary_value(*kind, input.evaluate_fma(sample)),
            Self::Binary { kind, left, right } => {
                binary_value(*kind, left.evaluate_fma(sample), right.evaluate_fma(sample))
            }
        }
    }

    /// Specialize the pointwise program into WGSL. The single seeded
    /// permutation is a runtime binding; constants are embedded by exact bits.
    pub fn wgsl(&self) -> String {
        let expression = self.emit("pt", "leaf_value", false);
        PERLIN_WGSL.replace(
            "output[i] = pt.amp * perlin(pt);",
            &format!("let leaf_value = perlin(pt);\n    output[i] = pt.amp * {expression};"),
        )
    }

    /// WGSL lowering paired with [`Self::evaluate_fma`].
    pub fn wgsl_fma(&self) -> String {
        let expression = self.emit("pt", "leaf_value", true);
        PERLIN_WGSL.replace(
            "output[i] = pt.amp * perlin(pt);",
            &format!("let leaf_value = perlin_fma(pt);\n    output[i] = pt.amp * {expression};"),
        )
    }

    /// WGSL lowering for a later composition pass whose Perlin leaf has been
    /// materialized by an earlier GPU dispatch.
    pub fn wgsl_with_precomputed_leaf(&self) -> String {
        self.wgsl().replace(
            "let leaf_value = perlin(pt);",
            "let leaf_value = leaf_values[i];",
        )
    }

    /// FMA lowering for graph composition after a separate Perlin FMA dispatch.
    pub fn wgsl_with_precomputed_leaf_fma(&self) -> String {
        self.wgsl_fma().replace(
            "let leaf_value = perlin_fma(pt);",
            "let leaf_value = leaf_values[i];",
        )
    }

    fn emit(&self, sample: &str, leaf: &str, use_fma: bool) -> String {
        match self {
            Self::Perlin(_) => {
                if sample == "pt" {
                    leaf.to_owned()
                } else if use_fma {
                    format!("perlin_fma({sample})")
                } else {
                    format!("perlin({sample})")
                }
            }
            Self::Constant(value) => format!("bitcast<f32>(0x{:08x}u)", value.to_bits()),
            Self::Lerp {
                alpha,
                first,
                second,
            }
            | Self::Blend {
                alpha,
                first,
                second,
            } => format!(
                "{}({}, {}, {})",
                if use_fma { "blend_fma" } else { "blend" },
                alpha.emit(sample, leaf, use_fma),
                first.emit(sample, leaf, use_fma),
                second.emit(sample, leaf, use_fma)
            ),
            Self::Spline {
                input,
                x0,
                x1,
                y0,
                y1,
                d0,
                d1,
            } => format!(
                "spline({}, {}, {}, {}, {}, {}, {})",
                input.emit(sample, leaf, use_fma),
                bits(*x0),
                bits(*x1),
                bits(*y0),
                bits(*y1),
                bits(*d0),
                bits(*d1)
            ),
            Self::Offset { input, delta } => {
                let shifted = format!(
                    "offset_sample({sample}, vec3<f32>({}, {}, {}))",
                    bits(delta[0]),
                    bits(delta[1]),
                    bits(delta[2])
                );
                input.emit(&shifted, "", use_fma)
            }
            Self::Add(left, right) => {
                format!(
                    "({} + {})",
                    left.emit(sample, leaf, use_fma),
                    right.emit(sample, leaf, use_fma)
                )
            }
            Self::Multiply(left, right) => {
                format!(
                    "({} * {})",
                    left.emit(sample, leaf, use_fma),
                    right.emit(sample, leaf, use_fma)
                )
            }
            Self::Clamp { input, min, max } => format!(
                "clamp_value({}, {}, {})",
                input.emit(sample, leaf, use_fma),
                bits(*min),
                bits(*max)
            ),
            Self::Unary { kind, input } => {
                let value = input.emit(sample, leaf, use_fma);
                match kind {
                    // `f32::abs` clears the sign bit even for NaN payloads; the
                    // bit trick matches it exactly where `abs()` is undefined.
                    UnaryKind::Abs => {
                        format!("bitcast<f32>(bitcast<u32>({value}) & 0x7fffffffu)")
                    }
                    UnaryKind::Negate => {
                        format!("bitcast<f32>(bitcast<u32>({value}) ^ 0x80000000u)")
                    }
                    UnaryKind::Square => format!("({value} * {value})"),
                    UnaryKind::Cube => format!("({value} * {value} * {value})"),
                    UnaryKind::HalfNegative => {
                        format!("select(({value} * 0.5), {value}, {value} > 0.0)")
                    }
                    UnaryKind::QuarterNegative => {
                        format!("select(({value} * 0.25), {value}, {value} > 0.0)")
                    }
                    UnaryKind::Reciprocal => format!("(1.0 / {value})"),
                    UnaryKind::Sign => format!(
                        "select(select(-1.0, 1.0, {value} > 0.0), {value}, {value} == 0.0 || is_nan_f({value}))"
                    ),
                    UnaryKind::Squeeze => format!("squeeze({value})"),
                }
            }
            Self::Binary { kind, left, right } => {
                let a = left.emit(sample, leaf, use_fma);
                let b = right.emit(sample, leaf, use_fma);
                match kind {
                    BinaryKind::Add => format!("({a} + {b})"),
                    BinaryKind::Subtract => format!("({a} - {b})"),
                    BinaryKind::Multiply => format!("({a} * {b})"),
                    BinaryKind::Divide => format!("({a} / {b})"),
                    BinaryKind::Minimum => format!("java_min_f({a}, {b})"),
                    BinaryKind::Maximum => format!("java_max_f({a}, {b})"),
                }
            }
        }
    }
}

fn bits(value: f32) -> String {
    format!("bitcast<f32>(0x{:08x}u)", value.to_bits())
}

fn offset_sample(mut sample: PerlinSample, delta: [f32; 3]) -> PerlinSample {
    for (axis, add) in delta.into_iter().enumerate() {
        let shifted = sample.offset[axis] + add;
        let cells = shifted.floor() as i32;
        sample.cell[axis] = sample.cell[axis].wrapping_add(cells);
        sample.offset[axis] = shifted - cells as f32;
    }
    sample.fade_y = sample.offset[1];
    sample
}

fn blend(alpha: f32, first: f32, second: f32) -> f32 {
    if alpha == 0.0 {
        first
    } else if alpha == 1.0 {
        second
    } else {
        first + alpha * (second - first)
    }
}

fn blend_fma(alpha: f32, first: f32, second: f32) -> f32 {
    if alpha == 0.0 {
        first
    } else if alpha == 1.0 {
        second
    } else {
        alpha.mul_add(second - first, first)
    }
}

/// Mirrors `UnaryOp::apply` (`arithmetic.rs`) for the unary kinds that are on
/// the exact tier. `Sqrt`/`Log`/`Power` differ by construction and are not part
/// of `UnaryKind`.
pub(crate) fn unary_value(kind: UnaryKind, value: f32) -> f32 {
    match kind {
        UnaryKind::Abs => value.abs(),
        UnaryKind::Negate => -value,
        UnaryKind::Square => value * value,
        UnaryKind::Cube => value * value * value,
        UnaryKind::HalfNegative => {
            if value > 0.0 {
                value
            } else {
                value * 0.5
            }
        }
        UnaryKind::QuarterNegative => {
            if value > 0.0 {
                value
            } else {
                value * 0.25
            }
        }
        UnaryKind::Reciprocal => 1.0 / value,
        UnaryKind::Sign => {
            if value == 0.0 || value.is_nan() {
                value
            } else {
                value.signum()
            }
        }
        UnaryKind::Squeeze => {
            let v = clamp_value(value, -1.0, 1.0);
            v / 2.0 - v * v * v / 24.0
        }
    }
}

/// Mirrors `BinaryOp::apply` (`arithmetic.rs`). The compiled scalar sampler's
/// early-outs (multiply/divide by a zero left operand, min/max bound hit) are
/// not part of this generic definition; they differ only for NaN operands or
/// signed zero on the early-out path (documented on the tier).
pub(crate) fn binary_value(kind: BinaryKind, left: f32, right: f32) -> f32 {
    match kind {
        BinaryKind::Add => left + right,
        BinaryKind::Subtract => left - right,
        BinaryKind::Multiply => left * right,
        BinaryKind::Divide => left / right,
        BinaryKind::Minimum => java_min(left, right),
        BinaryKind::Maximum => java_max(left, right),
    }
}

fn spline(x: f32, x0: f32, x1: f32, y0: f32, y1: f32, d0: f32, d1: f32) -> f32 {
    let width = x1 - x0;
    let alpha = (x - x0) / width;
    let delta = y1 - y0;
    let left = d0.mul_add(width, -delta);
    let right = (-d1).mul_add(width, delta);
    let linear = alpha.mul_add(delta, y0);
    let correction = alpha.mul_add(right - left, left);
    (alpha * (1.0 - alpha)).mul_add(correction, linear)
}

impl PerlinProgram {
    pub fn new(permutation: [u8; 256]) -> Self {
        Self { permutation }
    }

    pub fn from_permutation(permutation: &[u8; 256]) -> Self {
        Self::new(*permutation)
    }

    pub fn evaluate(&self, sample: PerlinSample) -> f32 {
        self.evaluate_with_fma(sample, false)
    }

    /// Evaluate with explicit fused lerp and fade operations, corresponding
    /// to Rust `mul_add` paired with WGSL `fma()`.
    pub fn evaluate_fma(&self, sample: PerlinSample) -> f32 {
        self.evaluate_with_fma(sample, true)
    }

    fn evaluate_with_fma(&self, sample: PerlinSample, use_fma: bool) -> f32 {
        let permute = |i: i32| i32::from(self.permutation[(i & 255) as usize]);
        let gradient = |i: i32| GRADIENTS[(permute(i) & 15) as usize];
        let [cx, cy, cz] = sample.cell;
        let p = permute(cx);
        let q = permute(cx.wrapping_add(1));
        let lower = permute(p.wrapping_add(cy));
        let upper = permute(p.wrapping_add(cy).wrapping_add(1));
        let lower_x = permute(q.wrapping_add(cy));
        let upper_x = permute(q.wrapping_add(cy).wrapping_add(1));
        let rows = [
            lower, lower_x, upper, upper_x, lower, lower_x, upper, upper_x,
        ];
        let [dx, dy, dz] = sample.offset;
        let fade = [
            smoothstep(dx, use_fma),
            smoothstep(sample.fade_y, use_fma),
            smoothstep(dz, use_fma),
        ];
        let mut values = [0.0; 8];
        for j in 0..8 {
            let g = gradient(rows[j].wrapping_add(cz).wrapping_add((j / 4) as i32));
            let px = dx - (j & 1) as f32;
            let py = dy - ((j >> 1) & 1) as f32;
            let pz = dz - ((j >> 2) & 1) as f32;
            values[j] = if use_fma {
                g[0].mul_add(px, g[1].mul_add(py, g[2] * pz))
            } else {
                g[0] * px + g[1] * py + g[2] * pz
            };
        }
        lerp3(fade[0], fade[1], fade[2], values, use_fma)
    }

    /// WGSL for an R1-only program.
    pub const fn wgsl() -> &'static str {
        PERLIN_WGSL
    }
}

fn smoothstep(v: f32, use_fma: bool) -> f32 {
    let inner = if use_fma {
        v.mul_add(6.0, -15.0)
    } else {
        v * 6.0 - 15.0
    };
    let outer = if use_fma {
        v.mul_add(inner, 10.0)
    } else {
        v * inner + 10.0
    };
    v * v * v * outer
}
fn lerp(delta: f32, start: f32, end: f32, use_fma: bool) -> f32 {
    if use_fma {
        delta.mul_add(end - start, start)
    } else {
        start + delta * (end - start)
    }
}
fn lerp2(a: f32, b: f32, v0: f32, v1: f32, v2: f32, v3: f32, use_fma: bool) -> f32 {
    lerp(
        b,
        lerp(a, v0, v1, use_fma),
        lerp(a, v2, v3, use_fma),
        use_fma,
    )
}
fn lerp3(fx: f32, fy: f32, fz: f32, v: [f32; 8], use_fma: bool) -> f32 {
    lerp(
        fz,
        lerp2(fx, fy, v[0], v[1], v[2], v[3], use_fma),
        lerp2(fx, fy, v[4], v[5], v[6], v[7], use_fma),
        use_fma,
    )
}

const PERLIN_WGSL: &str = r#"
struct Point {
    cell: vec4<i32>,
    off: vec4<f32>,
    fade_y: f32,
    amp: f32,
    pad: vec2<f32>,
}
@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> perm: array<u32>;
@group(0) @binding(3) var<storage, read> grads: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> leaf_values: array<f32>;

fn permute(i: i32) -> i32 { return i32(perm[i & 255]); }
fn gradient(i: i32) -> vec3<f32> {
    let g = grads[permute(i) & 15];
    return vec3<f32>(g.x, g.y, g.z);
}
fn fade(v: f32) -> f32 { return v * v * v * (v * (v * 6.0 - 15.0) + 10.0); }
fn lerp1(delta: f32, start: f32, end: f32) -> f32 { return start + delta * (end - start); }
fn lerp2(a: f32, b: f32, v0: f32, v1: f32, v2: f32, v3: f32) -> f32 {
    return lerp1(b, lerp1(a, v0, v1), lerp1(a, v2, v3));
}
fn lerp3(fx: f32, fy: f32, fz: f32, v: array<f32, 8>) -> f32 {
    return lerp1(fz, lerp2(fx, fy, v[0], v[1], v[2], v[3]),
                 lerp2(fx, fy, v[4], v[5], v[6], v[7]));
}
fn fade_fma(v: f32) -> f32 {
    let inner = fma(v, 6.0, -15.0);
    let outer = fma(v, inner, 10.0);
    return v * v * v * outer;
}
fn lerp1_fma(delta: f32, start: f32, end: f32) -> f32 {
    return fma(delta, end - start, start);
}
fn lerp2_fma(a: f32, b: f32, v0: f32, v1: f32, v2: f32, v3: f32) -> f32 {
    return lerp1_fma(b, lerp1_fma(a, v0, v1), lerp1_fma(a, v2, v3));
}
fn lerp3_fma(fx: f32, fy: f32, fz: f32, v: array<f32, 8>) -> f32 {
    return lerp1_fma(fz, lerp2_fma(fx, fy, v[0], v[1], v[2], v[3]),
                     lerp2_fma(fx, fy, v[4], v[5], v[6], v[7]));
}
fn is_nan_f(value: f32) -> bool {
    // WGSL has no isNan builtin; a NaN has all exponent bits set and a
    // non-zero mantissa.
    return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u;
}
fn java_min_f(a: f32, b: f32) -> f32 {
    if (is_nan_f(a) || is_nan_f(b)) { return bitcast<f32>(0x7fc00000u); }
    if (a == 0.0 && b == 0.0) {
        return bitcast<f32>(bitcast<u32>(a) | bitcast<u32>(b));
    }
    return select(b, a, a <= b);
}
fn java_max_f(a: f32, b: f32) -> f32 {
    if (is_nan_f(a) || is_nan_f(b)) { return bitcast<f32>(0x7fc00000u); }
    if (a == 0.0 && b == 0.0) {
        return bitcast<f32>(bitcast<u32>(a) & bitcast<u32>(b));
    }
    return select(b, a, a >= b);
}
fn clamp_value(value: f32, min: f32, max: f32) -> f32 {
    if (value < min) { return min; }
    return java_min_f(value, max);
}fn squeeze(value: f32) -> f32 {
    let v = clamp_value(value, -1.0, 1.0);
    return v / 2.0 - v * v * v / 24.0;
}
fn blend(alpha: f32, first: f32, second: f32) -> f32 {
    if (alpha == 0.0) { return first; }
    if (alpha == 1.0) { return second; }
    return first + alpha * (second - first);
}
fn blend_fma(alpha: f32, first: f32, second: f32) -> f32 {
    if (alpha == 0.0) { return first; }
    if (alpha == 1.0) { return second; }
    return fma(alpha, second - first, first);
}
fn spline(x: f32, x0: f32, x1: f32, y0: f32, y1: f32, d0: f32, d1: f32) -> f32 {
    let width = x1 - x0;
    let alpha = (x - x0) / width;
    let delta = y1 - y0;
    let left = fma(d0, width, -delta);
    let right = fma(-d1, width, delta);
    let linear = fma(alpha, delta, y0);
    let correction = fma(alpha, right - left, left);
    return fma(alpha * (1.0 - alpha), correction, linear);
}
fn offset_sample(pt: Point, delta: vec3<f32>) -> Point {
    var shifted = pt;
    let value = pt.off.xyz + delta;
    let cells = vec3<i32>(floor(value));
    shifted.cell = vec4<i32>(pt.cell.xyz + cells, pt.cell.w);
    shifted.off = vec4<f32>(value - vec3<f32>(cells), pt.off.w);
    shifted.fade_y = shifted.off.y;
    return shifted;
}
fn perlin(pt: Point) -> f32 {
    let cx = pt.cell.x; let cy = pt.cell.y; let cz = pt.cell.z;
    let p = permute(cx); let q = permute(cx + 1);
    let lower = permute(p + cy); let upper = permute(p + cy + 1);
    let lower_x = permute(q + cy); let upper_x = permute(q + cy + 1);
    var rows: array<i32, 8> = array<i32, 8>(lower, lower_x, upper, upper_x,
                                            lower, lower_x, upper, upper_x);
    var values: array<f32, 8>;
    let dx = pt.off.x; let dy = pt.off.y; let dz = pt.off.z;
    let fx = fade(dx); let fy = fade(pt.fade_y); let fz = fade(dz);
    for (var j: i32 = 0; j < 8; j = j + 1) {
        let g = gradient(rows[j] + cz + (j >> 2));
        let px = dx - f32(j & 1);
        let py = dy - f32((j >> 1) & 1);
        let pz = dz - f32((j >> 2) & 1);
        values[j] = fma(g.x, px, fma(g.y, py, g.z * pz));
    }
    return lerp3(fx, fy, fz, values);
}
fn perlin_fma(pt: Point) -> f32 {
    let cx = pt.cell.x; let cy = pt.cell.y; let cz = pt.cell.z;
    let p = permute(cx); let q = permute(cx + 1);
    let lower = permute(p + cy); let upper = permute(p + cy + 1);
    let lower_x = permute(q + cy); let upper_x = permute(q + cy + 1);
    var rows: array<i32, 8> = array<i32, 8>(lower, lower_x, upper, upper_x,
                                            lower, lower_x, upper, upper_x);
    var values: array<f32, 8>;
    let dx = pt.off.x; let dy = pt.off.y; let dz = pt.off.z;
    let fx = fade_fma(dx); let fy = fade_fma(pt.fade_y); let fz = fade_fma(dz);
    for (var j: i32 = 0; j < 8; j = j + 1) {
        let g = gradient(rows[j] + cz + (j >> 2));
        let px = dx - f32(j & 1);
        let py = dy - f32((j >> 1) & 1);
        let pz = dz - f32((j >> 2));
        values[j] = fma(g.x, px, fma(g.y, py, g.z * pz));
    }
    return lerp3_fma(fx, fy, fz, values);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= arrayLength(&points)) { return; }
    let pt = points[i];
    output[i] = pt.amp * perlin(pt);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{noise::PerlinNoise, random::LegacyRandom};

    #[test]
    fn evaluator_uses_existing_seeded_permutation_table() {
        let mut random = LegacyRandom::new(0x5eed_1234);
        let noise = PerlinNoise::new(&mut random);
        let program = PerlinProgram::from_permutation(noise.table().permutations());
        let point = [-12.375, 8.25, 0.5];
        let offsets = noise.table().offsets();
        let coords = std::array::from_fn::<_, 3, _>(|axis| {
            let shifted = crate::noise::wrap(point[axis]) + offsets[axis];
            let cell = shifted.floor() as i32;
            (cell, (shifted - f64::from(cell)) as f32)
        });
        let sample = PerlinSample {
            cell: coords.map(|(cell, _)| cell),
            offset: coords.map(|(_, fraction)| fraction),
            fade_y: coords[1].1,
        };
        assert_eq!(
            program.evaluate(sample).to_bits(),
            noise.get3(point[0], point[1], point[2]).to_bits()
        );
    }
}
