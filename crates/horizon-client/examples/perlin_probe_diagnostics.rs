use super::{GRADIENTS, Point};

fn permute(perm: &[u32; 256], i: i32) -> i32 {
    perm[(i & 255) as usize] as i32
}
fn gradient(perm: &[u32; 256], i: i32) -> [f32; 3] {
    GRADIENTS[(permute(perm, i) & 15) as usize]
}
fn lerp_cpu(delta: f32, start: f32, end: f32, fma: bool) -> f32 {
    if fma {
        delta.mul_add(end - start, start)
    } else {
        start + delta * (end - start)
    }
}
fn lerp2_cpu(a: f32, b: f32, v: [f32; 4], fma: bool) -> f32 {
    lerp_cpu(
        b,
        lerp_cpu(a, v[0], v[1], fma),
        lerp_cpu(a, v[2], v[3], fma),
        fma,
    )
}
fn lerp3_cpu(fade: [f32; 3], values: [f32; 8], fma: bool) -> f32 {
    lerp_cpu(
        fade[2],
        lerp2_cpu(fade[0], fade[1], values[..4].try_into().unwrap(), fma),
        lerp2_cpu(fade[0], fade[1], values[4..].try_into().unwrap(), fma),
        fma,
    )
}
fn smoothstep_cpu(v: f32, fma_inner: bool, fma_outer: bool) -> f32 {
    let a = if fma_inner {
        v.mul_add(6.0, -15.0)
    } else {
        v * 6.0 - 15.0
    };
    let b = if fma_outer {
        v.mul_add(a, 10.0)
    } else {
        v * a + 10.0
    };
    v * v * v * b
}
fn sample_cpu(
    pt: &Point,
    perm: &[u32; 256],
    fma_lerp: bool,
    fma_inner: bool,
    fma_outer: bool,
    assoc: u8,
) -> f32 {
    let [cx, cy, cz, _] = pt.cell;
    let p = permute(perm, cx);
    let q = permute(perm, cx + 1);
    let lower = permute(perm, p + cy);
    let upper = permute(perm, p + cy + 1);
    let lower_x = permute(perm, q + cy);
    let upper_x = permute(perm, q + cy + 1);
    let rows = [
        lower, lower_x, upper, upper_x, lower, lower_x, upper, upper_x,
    ];
    let [dx, dy, dz, _] = pt.off;
    let mut values = [0.0f32; 8];
    for j in 0..8 {
        let g = gradient(perm, rows[j] + cz + (j >> 2) as i32);
        let px = dx - (j & 1) as f32;
        let py = dy - ((j >> 1) & 1) as f32;
        let pz = dz - ((j >> 2) & 1) as f32;
        values[j] = match assoc {
            0 => g[0] * px + g[1] * py + g[2] * pz,
            1 => g[0] * px + (g[1] * py + g[2] * pz),
            _ => g[0] * px + g[2] * pz + g[1] * py,
        };
    }
    let fade = [
        smoothstep_cpu(dx, fma_inner, fma_outer),
        smoothstep_cpu(pt.fade_y, fma_inner, fma_outer),
        smoothstep_cpu(dz, fma_inner, fma_outer),
    ];
    pt.amp * lerp3_cpu(fade, values, fma_lerp)
}

pub(super) fn report_fma(points: &[Point], perm: &[u32; 256], gpu: &[f32], expected: &[f32]) {
    for (fma_lerp, fma_inner, fma_outer, assoc) in [
        (false, false, false, 0),
        (true, false, false, 0),
        (false, true, false, 0),
        (false, false, true, 0),
        (true, true, false, 0),
        (true, false, true, 0),
        (false, true, true, 0),
        (true, true, true, 0),
        (false, false, false, 1),
        (false, false, false, 2),
    ] {
        let mut vs_get3 = 0;
        let mut vs_gpu = 0;
        for ((point, gpu_value), expected_value) in points.iter().zip(gpu).zip(expected) {
            let cpu = sample_cpu(point, perm, fma_lerp, fma_inner, fma_outer, assoc);
            if cpu.to_bits() != expected_value.to_bits() {
                vs_get3 += 1;
            }
            if cpu.to_bits() != gpu_value.to_bits() {
                vs_gpu += 1;
            }
        }
        println!(
            "  CPU fma(lerp)={fma_lerp} fma(smooth inner)={fma_inner} fma(smooth outer)={fma_outer} assoc={assoc}: {vs_get3} vs get3, {vs_gpu} vs GPU"
        );
    }
}
