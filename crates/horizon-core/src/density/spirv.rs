//! Text SPIR-V emission for [`DensityProgram`].
//!
//! The point of this module is the *contract*, not convenience: every emitted
//! floating-point add/sub/mul/div carries `OpDecorate ... NoContraction`, and
//! every site where [`DensityProgram::evaluate_fma`] calls `mul_add` is an
//! explicit `GLSL.std.450 Fma`. No fast-math mode is ever emitted. The module
//! is assembled with `spirv-as --target-env vulkan1.3` and handed to a driver
//! through wgpu's SPIR-V passthrough, so no translator can drop them.
//!
//! One invocation per sample: the input buffer carries the already-narrowed
//! Perlin sample plus the block position (needed by `Gradient` and by scaled
//! noise leaves), and the output is `amp * evaluate_fma(sample)`.
//!
//! `program.rs` remains the single source of truth for semantics: this emitter
//! mirrors `evaluate_fma` case by case, and the probe compares the two on
//! 60,000 samples per rung.

use super::program::{BinaryKind, DensityProgram, UnaryKind};

/// Expression context: the sample the subexpression is evaluated at.
#[derive(Clone, Copy)]
struct Context {
    cell: u32,
    off: u32,
    fade_y: u32,
}

struct Emitter {
    decorations: Vec<String>,
    types: Vec<String>,
    body: Vec<String>,
    next: u32,
    f32_consts: std::collections::HashMap<u32, u32>,
    i32_consts: std::collections::HashMap<i32, u32>,
    u32_consts: std::collections::HashMap<u32, u32>,
    ids: Ids,
}

#[derive(Clone, Copy)]
struct Ids {
    void: u32,
    fn_ty: u32,
    bool_ty: u32,
    i32: u32,
    u32: u32,
    f32: u32,
    v3i: u32,
    v3u: u32,
    v4i: u32,
    v4f: u32,
    point: u32,
    rt_point: u32,
    rt_f: u32,
    rt_u: u32,
    rt_v4f: u32,
    buf_points: u32,
    buf_f: u32,
    buf_u: u32,
    buf_v4f: u32,
    p_points: u32,
    p_f: u32,
    p_u: u32,
    p_v4f: u32,
    p_pf32: u32,
    p_pu32: u32,
    p_pv4f: u32,
    p_pv4i: u32,
    p_pv3i: u32,
    p_in_v3u: u32,
    perm: u32,
    grads: u32,
    out: u32,
    points: u32,
    gid: u32,
    glsl: u32,
    main: u32,
}

impl Emitter {
    fn new() -> Self {
        let mut e = Emitter {
            decorations: Vec::new(),
            types: Vec::new(),
            body: Vec::new(),
            next: 1,
            f32_consts: Default::default(),
            i32_consts: Default::default(),
            u32_consts: Default::default(),
            ids: Ids {
                void: 0,
                fn_ty: 0,
                bool_ty: 0,
                i32: 0,
                u32: 0,
                f32: 0,
                v3i: 0,
                v3u: 0,
                v4i: 0,
                v4f: 0,
                point: 0,
                rt_point: 0,
                rt_f: 0,
                rt_u: 0,
                rt_v4f: 0,
                buf_points: 0,
                buf_f: 0,
                buf_u: 0,
                buf_v4f: 0,
                p_points: 0,
                p_f: 0,
                p_u: 0,
                p_v4f: 0,
                p_pf32: 0,
                p_pu32: 0,
                p_pv4f: 0,
                p_pv4i: 0,
                p_pv3i: 0,
                p_in_v3u: 0,
                perm: 0,
                grads: 0,
                out: 0,
                points: 0,
                gid: 0,
                glsl: 0,
                main: 0,
            },
        };
        let names = [
            "void",
            "fn_ty",
            "bool_ty",
            "i32",
            "u32",
            "f32",
            "v3i",
            "v3u",
            "v4i",
            "v4f",
            "point",
            "rt_point",
            "rt_f",
            "rt_u",
            "rt_v4f",
            "buf_points",
            "buf_f",
            "buf_u",
            "buf_v4f",
            "p_points",
            "p_f",
            "p_u",
            "p_v4f",
            "p_pf32",
            "p_pu32",
            "p_pv4f",
            "p_pv4i",
            "p_pv3i",
            "p_in_v3u",
            "perm",
            "grads",
            "out",
            "points",
            "gid",
            "glsl",
            "main",
        ];
        for name in names {
            let id = e.id();
            match name {
                "void" => e.ids.void = id,
                "fn_ty" => e.ids.fn_ty = id,
                "bool_ty" => e.ids.bool_ty = id,
                "i32" => e.ids.i32 = id,
                "u32" => e.ids.u32 = id,
                "f32" => e.ids.f32 = id,
                "v3i" => e.ids.v3i = id,
                "v3u" => e.ids.v3u = id,
                "v4i" => e.ids.v4i = id,
                "v4f" => e.ids.v4f = id,
                "point" => e.ids.point = id,
                "rt_point" => e.ids.rt_point = id,
                "rt_f" => e.ids.rt_f = id,
                "rt_u" => e.ids.rt_u = id,
                "rt_v4f" => e.ids.rt_v4f = id,
                "buf_points" => e.ids.buf_points = id,
                "buf_f" => e.ids.buf_f = id,
                "buf_u" => e.ids.buf_u = id,
                "buf_v4f" => e.ids.buf_v4f = id,
                "p_points" => e.ids.p_points = id,
                "p_f" => e.ids.p_f = id,
                "p_u" => e.ids.p_u = id,
                "p_v4f" => e.ids.p_v4f = id,
                "p_pf32" => e.ids.p_pf32 = id,
                "p_pu32" => e.ids.p_pu32 = id,
                "p_pv4f" => e.ids.p_pv4f = id,
                "p_pv4i" => e.ids.p_pv4i = id,
                "p_pv3i" => e.ids.p_pv3i = id,
                "p_in_v3u" => e.ids.p_in_v3u = id,
                "perm" => e.ids.perm = id,
                "grads" => e.ids.grads = id,
                "out" => e.ids.out = id,
                "points" => e.ids.points = id,
                "gid" => e.ids.gid = id,
                "glsl" => e.ids.glsl = id,
                "main" => e.ids.main = id,
                _ => unreachable!(),
            }
        }
        e.header();
        e
    }

    fn id(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }

    fn header(&mut self) {
        let i = self.ids;
        let t = &mut self.types;
        t.push(format!("%{} = OpTypeVoid", i.void));
        t.push(format!("%{} = OpTypeBool", i.bool_ty));
        t.push(format!("%{} = OpTypeInt 32 1", i.i32));
        t.push(format!("%{} = OpTypeInt 32 0", i.u32));
        t.push(format!("%{} = OpTypeFloat 32", i.f32));
        t.push(format!("%{} = OpTypeFunction %{}", i.fn_ty, i.void));
        t.push(format!("%{} = OpTypeVector %{} 3", i.v3i, i.i32));
        t.push(format!("%{} = OpTypeVector %{} 3", i.v3u, i.u32));
        t.push(format!("%{} = OpTypeVector %{} 4", i.v4i, i.i32));
        t.push(format!("%{} = OpTypeVector %{} 4", i.v4f, i.f32));
        t.push(format!(
            "%{} = OpTypeStruct %{} %{} %{} %{} %{} %{}",
            i.point, i.v4i, i.v4f, i.v3i, i.f32, i.f32, i.f32
        ));
        t.push(format!("%{} = OpTypeRuntimeArray %{}", i.rt_point, i.point));
        t.push(format!("%{} = OpTypeRuntimeArray %{}", i.rt_f, i.f32));
        t.push(format!("%{} = OpTypeRuntimeArray %{}", i.rt_u, i.u32));
        t.push(format!("%{} = OpTypeRuntimeArray %{}", i.rt_v4f, i.v4f));
        t.push(format!("%{} = OpTypeStruct %{}", i.buf_points, i.rt_point));
        t.push(format!("%{} = OpTypeStruct %{}", i.buf_f, i.rt_f));
        t.push(format!("%{} = OpTypeStruct %{}", i.buf_u, i.rt_u));
        t.push(format!("%{} = OpTypeStruct %{}", i.buf_v4f, i.rt_v4f));
        t.push(format!(
            "%{} = OpTypePointer StorageBuffer %{}",
            i.p_points, i.buf_points
        ));
        t.push(format!(
            "%{} = OpTypePointer StorageBuffer %{}",
            i.p_f, i.buf_f
        ));
        t.push(format!(
            "%{} = OpTypePointer StorageBuffer %{}",
            i.p_u, i.buf_u
        ));
        t.push(format!(
            "%{} = OpTypePointer StorageBuffer %{}",
            i.p_v4f, i.buf_v4f
        ));
        t.push(format!(
            "%{} = OpTypePointer StorageBuffer %{}",
            i.p_pf32, i.f32
        ));
        t.push(format!(
            "%{} = OpTypePointer StorageBuffer %{}",
            i.p_pu32, i.u32
        ));
        t.push(format!(
            "%{} = OpTypePointer StorageBuffer %{}",
            i.p_pv4f, i.v4f
        ));
        t.push(format!(
            "%{} = OpTypePointer StorageBuffer %{}",
            i.p_pv4i, i.v4i
        ));
        t.push(format!(
            "%{} = OpTypePointer StorageBuffer %{}",
            i.p_pv3i, i.v3i
        ));
        t.push(format!("%{} = OpTypePointer Input %{}", i.p_in_v3u, i.v3u));
        t.push(format!("%{} = OpVariable %{} StorageBuffer", i.perm, i.p_u));
        t.push(format!(
            "%{} = OpVariable %{} StorageBuffer",
            i.grads, i.p_v4f
        ));
        t.push(format!("%{} = OpVariable %{} StorageBuffer", i.out, i.p_f));
        t.push(format!(
            "%{} = OpVariable %{} StorageBuffer",
            i.points, i.p_points
        ));
        t.push(format!("%{} = OpVariable %{} Input", i.gid, i.p_in_v3u));

        let d = &mut self.decorations;
        d.push(format!("OpDecorate %{} Block", i.buf_points));
        d.push(format!("OpMemberDecorate %{} 0 Offset 0", i.buf_points));
        d.push(format!("OpDecorate %{} ArrayStride 64", i.rt_point));
        d.push(format!("OpDecorate %{} Block", i.buf_f));
        d.push(format!("OpMemberDecorate %{} 0 Offset 0", i.buf_f));
        d.push(format!("OpDecorate %{} ArrayStride 4", i.rt_f));
        d.push(format!("OpDecorate %{} Block", i.buf_u));
        d.push(format!("OpMemberDecorate %{} 0 Offset 0", i.buf_u));
        d.push(format!("OpDecorate %{} ArrayStride 4", i.rt_u));
        d.push(format!("OpDecorate %{} Block", i.buf_v4f));
        d.push(format!("OpMemberDecorate %{} 0 Offset 0", i.buf_v4f));
        d.push(format!("OpDecorate %{} ArrayStride 16", i.rt_v4f));
        d.push(format!("OpMemberDecorate %{} 0 Offset 0", i.point));
        d.push(format!("OpMemberDecorate %{} 1 Offset 16", i.point));
        d.push(format!("OpMemberDecorate %{} 2 Offset 32", i.point));
        d.push(format!("OpMemberDecorate %{} 3 Offset 44", i.point));
        d.push(format!("OpMemberDecorate %{} 4 Offset 48", i.point));
        d.push(format!("OpMemberDecorate %{} 5 Offset 52", i.point));
        d.push(format!("OpDecorate %{} DescriptorSet 0", i.points));
        d.push(format!("OpDecorate %{} Binding 0", i.points));
        d.push(format!("OpDecorate %{} NonWritable", i.points));
        d.push(format!("OpDecorate %{} DescriptorSet 0", i.out));
        d.push(format!("OpDecorate %{} Binding 1", i.out));
        d.push(format!("OpDecorate %{} DescriptorSet 0", i.perm));
        d.push(format!("OpDecorate %{} Binding 2", i.perm));
        d.push(format!("OpDecorate %{} NonWritable", i.perm));
        d.push(format!("OpDecorate %{} DescriptorSet 0", i.grads));
        d.push(format!("OpDecorate %{} Binding 3", i.grads));
        d.push(format!("OpDecorate %{} NonWritable", i.grads));
        d.push(format!("OpDecorate %{} BuiltIn GlobalInvocationId", i.gid));
    }

    fn i32_const(&mut self, value: i32) -> u32 {
        if let Some(&id) = self.i32_consts.get(&value) {
            return id;
        }
        let id = self.id();
        self.types
            .push(format!("%{id} = OpConstant %{} {value}", self.ids.i32));
        self.i32_consts.insert(value, id);
        id
    }

    fn u32_const(&mut self, value: u32) -> u32 {
        if let Some(&id) = self.u32_consts.get(&value) {
            return id;
        }
        let id = self.id();
        self.types
            .push(format!("%{id} = OpConstant %{} {value}", self.ids.u32));
        self.u32_consts.insert(value, id);
        id
    }

    /// Constants carry their exact bit pattern, so NaN payloads and signed
    /// zeros survive the text round trip.
    fn f32_const(&mut self, value: f32) -> u32 {
        let bits = value.to_bits();
        if let Some(&id) = self.f32_consts.get(&bits) {
            return id;
        }
        let id = self.id();
        self.types.push(format!(
            "%{id} = OpConstant %{} !0x{bits:08x}",
            self.ids.f32
        ));
        self.f32_consts.insert(bits, id);
        id
    }

    fn op(&mut self, text: String) -> u32 {
        let id = self.id();
        self.body.push(format!("%{id} = {text}"));
        id
    }

    fn float_op(&mut self, opcode: &str, a: u32, b: u32) -> u32 {
        let f = self.ids.f32;
        let id = self.op(format!("{opcode} %{f} %{a} %{b}"));
        self.decorations
            .push(format!("OpDecorate %{id} NoContraction"));
        id
    }

    fn fadd(&mut self, a: u32, b: u32) -> u32 {
        self.float_op("OpFAdd", a, b)
    }

    fn fsub(&mut self, a: u32, b: u32) -> u32 {
        self.float_op("OpFSub", a, b)
    }

    fn fmul(&mut self, a: u32, b: u32) -> u32 {
        self.float_op("OpFMul", a, b)
    }

    fn fdiv(&mut self, a: u32, b: u32) -> u32 {
        self.float_op("OpFDiv", a, b)
    }

    fn fneg(&mut self, a: u32) -> u32 {
        let f = self.ids.f32;
        self.op(format!("OpFNegate %{f} %{a}"))
    }

    fn fma(&mut self, a: u32, b: u32, c: u32) -> u32 {
        let f = self.ids.f32;
        let g = self.ids.glsl;
        self.op(format!("OpExtInst %{f} %{g} Fma %{a} %{b} %{c}"))
    }

    fn ext(&mut self, name: &str, a: u32) -> u32 {
        let f = self.ids.f32;
        let g = self.ids.glsl;
        self.op(format!("OpExtInst %{f} %{g} {name} %{a}"))
    }

    fn cmp_f(&mut self, opcode: &str, left: u32, right: u32) -> u32 {
        let bool_ty = self.ids.bool_ty;
        self.op(format!("{opcode} %{bool_ty} %{left} %{right}"))
    }

    fn logical(&mut self, opcode: &str, left: u32, right: u32) -> u32 {
        let bool_ty = self.ids.bool_ty;
        self.op(format!("{opcode} %{bool_ty} %{left} %{right}"))
    }

    fn select_f(&mut self, cond: u32, a: u32, b: u32) -> u32 {
        let f = self.ids.f32;
        self.op(format!("OpSelect %{f} %{cond} %{a} %{b}"))
    }

    fn is_nan(&mut self, a: u32) -> u32 {
        let b = self.ids.bool_ty;
        self.op(format!("OpIsNan %{b} %{a}"))
    }

    fn bitcast_u(&mut self, a: u32) -> u32 {
        let u = self.ids.u32;
        self.op(format!("OpBitcast %{u} %{a}"))
    }

    fn bitcast_f(&mut self, a: u32) -> u32 {
        let f = self.ids.f32;
        self.op(format!("OpBitcast %{f} %{a}"))
    }

    fn iadd(&mut self, a: u32, b: u32) -> u32 {
        let i = self.ids.i32;
        self.op(format!("OpIAdd %{i} %{a} %{b}"))
    }

    fn iand(&mut self, a: u32, b: u32) -> u32 {
        let i = self.ids.i32;
        self.op(format!("OpBitwiseAnd %{i} %{a} %{b}"))
    }

    fn index1(&mut self, vector: u32, index: u32) -> u32 {
        let ty = self.ids.i32;
        self.op(format!("OpCompositeExtract %{ty} %{vector} {index}"))
    }

    fn indexf(&mut self, vector: u32, index: u32) -> u32 {
        let ty = self.ids.f32;
        self.op(format!("OpCompositeExtract %{ty} %{vector} {index}"))
    }

    /// `java_min` from `arithmetic.rs`: NaN propagates, and when both operands
    /// are zero the result keeps either sign bit (`a | b`).
    fn java_min(&mut self, a: u32, b: u32) -> u32 {
        let nan = self.f32_const(f32::NAN);
        let zero = self.f32_const(0.0);
        let a_nan = self.is_nan(a);
        let b_nan = self.is_nan(b);
        let any_nan = self.logical("OpLogicalOr", a_nan, b_nan);
        let a_zero = self.cmp_f("OpFOrdEqual", a, zero);
        let b_zero = self.cmp_f("OpFOrdEqual", b, zero);
        let both_zero = self.logical("OpLogicalAnd", a_zero, b_zero);
        let bits = {
            let u = self.ids.u32;
            let ab = self.bitcast_u(a);
            let bb = self.bitcast_u(b);
            self.op(format!("OpBitwiseOr %{u} %{ab} %{bb}"))
        };
        let zeros = self.bitcast_f(bits);
        let less_equal = self.cmp_f("OpFOrdLessThanEqual", a, b);
        let ordered = self.select_f(less_equal, a, b);
        let finite = self.select_f(both_zero, zeros, ordered);
        self.select_f(any_nan, nan, finite)
    }

    /// `java_max`: mirror of the above with `a & b` for the zero case.
    fn java_max(&mut self, a: u32, b: u32) -> u32 {
        let nan = self.f32_const(f32::NAN);
        let zero = self.f32_const(0.0);
        let a_nan = self.is_nan(a);
        let b_nan = self.is_nan(b);
        let any_nan = self.logical("OpLogicalOr", a_nan, b_nan);
        let a_zero = self.cmp_f("OpFOrdEqual", a, zero);
        let b_zero = self.cmp_f("OpFOrdEqual", b, zero);
        let both_zero = self.logical("OpLogicalAnd", a_zero, b_zero);
        let bits = {
            let u = self.ids.u32;
            let ab = self.bitcast_u(a);
            let bb = self.bitcast_u(b);
            self.op(format!("OpBitwiseAnd %{u} %{ab} %{bb}"))
        };
        let zeros = self.bitcast_f(bits);
        let greater_equal = self.cmp_f("OpFOrdGreaterThanEqual", a, b);
        let ordered = self.select_f(greater_equal, a, b);
        let finite = self.select_f(both_zero, zeros, ordered);
        self.select_f(any_nan, nan, finite)
    }

    /// `clamp(value, min, max)` from `arithmetic.rs`.
    fn clamp_value(&mut self, value: u32, min: f32, max: f32) -> u32 {
        let min_id = self.f32_const(min);
        let max_id = self.f32_const(max);
        let below = self.cmp_f("OpFOrdLessThan", value, min_id);
        let upper = self.java_min(value, max_id);
        self.select_f(below, min_id, upper)
    }

    /// `fade_fma`: `v*v*v * fma(v, fma(v, 6, -15), 10)`.
    fn fade(&mut self, v: u32) -> u32 {
        let six = self.f32_const(6.0);
        let neg15 = self.f32_const(-15.0);
        let ten = self.f32_const(10.0);
        let inner = self.fma(v, six, neg15);
        let outer = self.fma(v, inner, ten);
        let q1 = self.fmul(v, v);
        let q2 = self.fmul(q1, v);
        self.fmul(q2, outer)
    }

    /// `lerp_fma(delta, start, end) = fma(delta, end - start, start)`.
    fn lerp(&mut self, delta: u32, start: u32, end: u32) -> u32 {
        let diff = self.fsub(end, start);
        self.fma(delta, diff, start)
    }

    /// `blend_fma` from `program.rs`, including its exact-endpoint branches.
    fn blend(&mut self, alpha: u32, first: u32, second: u32) -> u32 {
        let diff = self.fsub(second, first);
        let joined = self.fma(alpha, diff, first);
        let zero = self.f32_const(0.0);
        let one = self.f32_const(1.0);
        let at_zero = self.cmp_f("OpFOrdEqual", alpha, zero);
        let at_one = self.cmp_f("OpFOrdEqual", alpha, one);
        let with_one = self.select_f(at_one, second, joined);
        self.select_f(at_zero, first, with_one)
    }

    fn permute(&mut self, index: u32) -> u32 {
        let mask = self.i32_const(255);
        let masked = self.iand(index, mask);
        let zero = self.i32_const(0);
        let (perm, ty) = (self.ids.perm, self.ids.p_pu32);
        let elem = self.op(format!("OpAccessChain %{ty} %{perm} %{zero} %{masked}"));
        let u = self.ids.u32;
        let raw = self.op(format!("OpLoad %{u} %{elem}"));
        let i = self.ids.i32;
        self.op(format!("OpBitcast %{i} %{raw}"))
    }

    fn gdot(&mut self, row: u32, px: u32, py: u32, pz: u32) -> u32 {
        let fifteen = self.i32_const(15);
        let p = self.permute(row);
        let idx = self.iand(p, fifteen);
        let zero = self.i32_const(0);
        let (grads, v4f, ty) = (self.ids.grads, self.ids.v4f, self.ids.p_pv4f);
        let elem = self.op(format!("OpAccessChain %{ty} %{grads} %{zero} %{idx}"));
        let g = self.op(format!("OpLoad %{v4f} %{elem}"));
        let gx = self.indexf(g, 0);
        let gy = self.indexf(g, 1);
        let gz = self.indexf(g, 2);
        let t = self.fmul(gz, pz);
        let w = self.fma(gy, py, t);
        self.fma(gx, px, w)
    }

    /// One seeded Perlin leaf, inlined, matching `PerlinProgram::evaluate_fma`.
    fn perlin(&mut self, ctx: Context) -> u32 {
        let cx = self.index1(ctx.cell, 0);
        let cy = self.index1(ctx.cell, 1);
        let cz = self.index1(ctx.cell, 2);
        let dx = self.indexf(ctx.off, 0);
        let dy = self.indexf(ctx.off, 1);
        let dz = self.indexf(ctx.off, 2);
        let one = self.i32_const(1);
        let p = self.permute(cx);
        let cx1 = self.iadd(cx, one);
        let q = self.permute(cx1);
        let p_cy = self.iadd(p, cy);
        let q_cy = self.iadd(q, cy);
        let p_cy1 = self.iadd(p_cy, one);
        let q_cy1 = self.iadd(q_cy, one);
        let lower = self.permute(p_cy);
        let upper = self.permute(p_cy1);
        let lower_x = self.permute(q_cy);
        let upper_x = self.permute(q_cy1);
        let rows = [
            lower, lower_x, upper, upper_x, lower, lower_x, upper, upper_x,
        ];
        let onef = self.f32_const(1.0);
        let dx1 = self.fsub(dx, onef);
        let dy1 = self.fsub(dy, onef);
        let dz1 = self.fsub(dz, onef);
        let mut values = [0u32; 8];
        for (j, row) in rows.iter().enumerate() {
            let cz2 = if j >= 4 { self.iadd(cz, one) } else { cz };
            let g = self.iadd(*row, cz2);
            let px = if j & 1 != 0 { dx1 } else { dx };
            let py = if j & 2 != 0 { dy1 } else { dy };
            let pz = if j & 4 != 0 { dz1 } else { dz };
            values[j] = self.gdot(g, px, py, pz);
        }
        let fx = self.fade(dx);
        let fy = self.fade(ctx.fade_y);
        let fz = self.fade(dz);
        let a0 = self.lerp(fx, values[0], values[1]);
        let a1 = self.lerp(fx, values[2], values[3]);
        let b0 = self.lerp(fx, values[4], values[5]);
        let b1 = self.lerp(fx, values[6], values[7]);
        let c0 = self.lerp(fy, a0, a1);
        let c1 = self.lerp(fy, b0, b1);
        self.lerp(fz, c0, c1)
    }

    fn emit(&mut self, ctx: Context, program: &DensityProgram) -> u32 {
        match program {
            DensityProgram::Perlin(_) => self.perlin(ctx),
            DensityProgram::Constant(value) => self.f32_const(*value),
            DensityProgram::Lerp {
                alpha,
                first,
                second,
            }
            | DensityProgram::Blend {
                alpha,
                first,
                second,
            } => {
                let a = self.emit(ctx, alpha);
                let f = self.emit(ctx, first);
                let s = self.emit(ctx, second);
                self.blend(a, f, s)
            }
            DensityProgram::Spline {
                input,
                x0,
                x1,
                y0,
                y1,
                d0,
                d1,
            } => {
                let x = self.emit(ctx, input);
                let x0c = self.f32_const(*x0);
                let x1c = self.f32_const(*x1);
                let y0c = self.f32_const(*y0);
                let y1c = self.f32_const(*y1);
                let d0c = self.f32_const(*d0);
                let d1c = self.f32_const(*d1);
                let width = self.fsub(x1c, x0c);
                let num = self.fsub(x, x0c);
                let alpha = self.fdiv(num, width);
                let delta = self.fsub(y1c, y0c);
                let neg_delta = self.fneg(delta);
                let left = self.fma(d0c, width, neg_delta);
                let neg_d1 = self.fneg(d1c);
                let right = self.fma(neg_d1, width, delta);
                let linear = self.fma(alpha, delta, y0c);
                let rl = self.fsub(right, left);
                let correction = self.fma(alpha, rl, left);
                let one = self.f32_const(1.0);
                let one_minus = self.fsub(one, alpha);
                let t = self.fmul(alpha, one_minus);
                self.fma(t, correction, linear)
            }
            DensityProgram::Offset { input, delta } => {
                // `offset_sample` from `program.rs`: narrow the shifted
                // coordinate back into a cell and a fraction.
                let x = self.indexf(ctx.off, 0);
                let y = self.indexf(ctx.off, 1);
                let z = self.indexf(ctx.off, 2);
                let dx = self.f32_const(delta[0]);
                let dy = self.f32_const(delta[1]);
                let dz = self.f32_const(delta[2]);
                let sx = self.fadd(x, dx);
                let sy = self.fadd(y, dy);
                let sz = self.fadd(z, dz);
                let fx = self.ext("Floor", sx);
                let fy = self.ext("Floor", sy);
                let fz = self.ext("Floor", sz);
                let i = self.ids.i32;
                let cix = self.op(format!("OpConvertFToS %{i} %{fx}"));
                let ciy = self.op(format!("OpConvertFToS %{i} %{fy}"));
                let ciz = self.op(format!("OpConvertFToS %{i} %{fz}"));
                let f = self.ids.f32;
                let cfx = self.op(format!("OpConvertSToF %{f} %{cix}"));
                let cfy = self.op(format!("OpConvertSToF %{f} %{ciy}"));
                let cfz = self.op(format!("OpConvertSToF %{f} %{ciz}"));
                let ox = self.fsub(sx, cfx);
                let oy = self.fsub(sy, cfy);
                let oz = self.fsub(sz, cfz);
                let cell = {
                    let base_x = self.index1(ctx.cell, 0);
                    let base_y = self.index1(ctx.cell, 1);
                    let base_z = self.index1(ctx.cell, 2);
                    let cx = self.iadd(base_x, cix);
                    let cy = self.iadd(base_y, ciy);
                    let cz = self.iadd(base_z, ciz);
                    self.composite3_i(cx, cy, cz)
                };
                let off = self.composite3_f(ox, oy, oz);
                self.emit(
                    Context {
                        cell,
                        off,
                        fade_y: oy,
                    },
                    input,
                )
            }
            DensityProgram::Add(left, right) => {
                let a = self.emit(ctx, left);
                let b = self.emit(ctx, right);
                self.fadd(a, b)
            }
            DensityProgram::Multiply(left, right) => {
                let a = self.emit(ctx, left);
                let b = self.emit(ctx, right);
                self.fmul(a, b)
            }
            DensityProgram::Clamp { input, min, max } => {
                let value = self.emit(ctx, input);
                self.clamp_value(value, *min, *max)
            }
            DensityProgram::Unary { kind, input } => {
                let value = self.emit(ctx, input);
                self.unary(*kind, value)
            }
            DensityProgram::Binary { kind, left, right } => {
                let a = self.emit(ctx, left);
                let b = self.emit(ctx, right);
                self.binary(*kind, a, b)
            }
        }
    }

    fn composite3_i(&mut self, x: u32, y: u32, z: u32) -> u32 {
        let ty = self.ids.v3i;
        let v = self.op(format!("OpCompositeConstruct %{ty} %{x} %{y} %{z}"));
        let v4 = self.ids.v4i;
        let w = self.i32_const(0);
        self.op(format!("OpCompositeConstruct %{v4} %{v} %{w}"))
    }

    fn composite3_f(&mut self, x: u32, y: u32, z: u32) -> u32 {
        let ty = self.ids.v4f;
        let w = self.f32_const(0.0);
        self.op(format!("OpCompositeConstruct %{ty} %{x} %{y} %{z} %{w}"))
    }

    fn unary(&mut self, kind: UnaryKind, value: u32) -> u32 {
        match kind {
            // `f32::abs`/`f32::neg` are sign-bit operations even for NaN;
            // FAbs/FNegate leave NaN payloads unspecified, so use bit ops.
            UnaryKind::Abs => {
                let (u, f) = (self.ids.u32, self.ids.f32);
                let bits = self.bitcast_u(value);
                let mask = self.u32_const(0x7fff_ffff);
                let raw = self.op(format!("OpBitwiseAnd %{u} %{bits} %{mask}"));
                self.op(format!("OpBitcast %{f} %{raw}"))
            }
            UnaryKind::Negate => {
                let (u, f) = (self.ids.u32, self.ids.f32);
                let bits = self.bitcast_u(value);
                let mask = self.u32_const(0x8000_0000);
                let raw = self.op(format!("OpBitwiseXor %{u} %{bits} %{mask}"));
                self.op(format!("OpBitcast %{f} %{raw}"))
            }
            UnaryKind::Square => self.fmul(value, value),
            UnaryKind::Cube => {
                let sq = self.fmul(value, value);
                self.fmul(sq, value)
            }
            UnaryKind::HalfNegative => self.leaky_relu(value, 0.5),
            UnaryKind::QuarterNegative => self.leaky_relu(value, 0.25),
            UnaryKind::Reciprocal => {
                let one = self.f32_const(1.0);
                self.fdiv(one, value)
            }
            UnaryKind::Sign => {
                let zero = self.f32_const(0.0);
                let is_zero = self.cmp_f("OpFOrdEqual", value, zero);
                let is_nan = self.is_nan(value);
                let keep = self.logical("OpLogicalOr", is_zero, is_nan);
                let positive = self.cmp_f("OpFOrdGreaterThan", value, zero);
                let one = self.f32_const(1.0);
                let neg_one = self.f32_const(-1.0);
                let signed = self.select_f(positive, one, neg_one);
                self.select_f(keep, value, signed)
            }
            UnaryKind::Squeeze => {
                let v = self.clamp_value(value, -1.0, 1.0);
                let two = self.f32_const(2.0);
                let twentyfour = self.f32_const(24.0);
                let half = self.fdiv(v, two);
                let sq = self.fmul(v, v);
                let cube = self.fmul(sq, v);
                let scaled = self.fdiv(cube, twentyfour);
                self.fsub(half, scaled)
            }
        }
    }

    fn leaky_relu(&mut self, value: u32, factor: f32) -> u32 {
        let zero = self.f32_const(0.0);
        let factor = self.f32_const(factor);
        let positive = self.cmp_f("OpFOrdGreaterThan", value, zero);
        let scaled = self.fmul(value, factor);
        self.select_f(positive, value, scaled)
    }

    fn binary(&mut self, kind: BinaryKind, a: u32, b: u32) -> u32 {
        match kind {
            BinaryKind::Add => self.fadd(a, b),
            BinaryKind::Subtract => self.fsub(a, b),
            BinaryKind::Multiply => self.fmul(a, b),
            BinaryKind::Divide => self.fdiv(a, b),
            BinaryKind::Minimum => self.java_min(a, b),
            BinaryKind::Maximum => self.java_max(a, b),
        }
    }
}

impl DensityProgram {
    /// Emit a complete compute module: one invocation per sample, output row
    /// `i` is `amp * program(sample i)`, with the explicit-FMA /
    /// no-contraction contract applied to every float operation.
    pub fn spirv(&self, _label: &str) -> String {
        let mut e = Emitter::new();
        let i = e.ids;
        let main_label = e.id();

        // Common prologue.
        e.body.push(format!("%{main_label} = OpLabel"));
        let gidv = {
            let ty = i.v3u;
            let gid = i.gid;
            e.op(format!("OpLoad %{ty} %{gid}"))
        };
        let ix = {
            let ty = i.u32;
            e.op(format!("OpCompositeExtract %{ty} %{gidv} 0"))
        };
        let inb = {
            let b = i.bool_ty;
            let n = e.u32_const(60_000);
            e.op(format!("OpULessThan %{b} %{ix} %{n}"))
        };
        let merge = e.id();
        let body_label = e.id();
        e.body.push(format!("OpSelectionMerge %{merge} None"));
        e.body
            .push(format!("OpBranchConditional %{inb} %{body_label} %{merge}"));
        e.body.push(format!("%{body_label} = OpLabel"));

        let zero_i = e.i32_const(0);
        let one_i = e.i32_const(1);
        let two_i = e.i32_const(2);
        let three_i = e.i32_const(3);
        let four_i = e.i32_const(4);
        let pti = {
            let ty = i.i32;
            e.op(format!("OpBitcast %{ty} %{ix}"))
        };
        let load_vec = |e: &mut Emitter, member: u32, ty: u32, ptr_ty: u32| {
            let points = e.ids.points;
            let ptr = e.op(format!(
                "OpAccessChain %{ptr_ty} %{points} %{zero_i} %{pti} %{member}"
            ));
            e.op(format!("OpLoad %{ty} %{ptr}"))
        };
        let cell = load_vec(&mut e, zero_i, i.v4i, i.p_pv4i);
        let off = load_vec(&mut e, one_i, i.v4f, i.p_pv4f);
        let block = load_vec(&mut e, two_i, i.v3i, i.p_pv3i);
        let fade_y = load_vec(&mut e, three_i, i.f32, i.p_pf32);
        let amp = load_vec(&mut e, four_i, i.f32, i.p_pf32);
        let _ = block;

        let ctx = Context { cell, off, fade_y };
        let value = e.emit(ctx, self);
        let scaled = e.fmul(amp, value);
        let out_ptr = {
            let (out, ty) = (i.out, i.p_pf32);
            e.op(format!("OpAccessChain %{ty} %{out} %{zero_i} %{ix}"))
        };
        e.body.push(format!("OpStore %{out_ptr} %{scaled}"));
        e.body.push(format!("OpBranch %{merge}"));
        e.body.push(format!("%{merge} = OpLabel"));
        e.body.push("OpReturn".to_owned());
        e.body.push("OpFunctionEnd".to_owned());

        let mut out = String::new();
        out.push_str("; Emitted by DensityProgram::spirv\n");
        out.push_str(
            "; Contract: every float add/sub/mul/div carries NoContraction; mul_add sites are Fma.\n",
        );
        out.push_str("OpCapability Shader\n");
        out.push_str(&format!("%{} = OpExtInstImport \"GLSL.std.450\"\n", i.glsl));
        out.push_str("OpMemoryModel Logical GLSL450\n");
        out.push_str(&format!(
            "OpEntryPoint GLCompute %{} \"main\" %{} %{} %{} %{} %{}\n",
            i.main, i.gid, i.points, i.out, i.perm, i.grads
        ));
        out.push_str(&format!("OpExecutionMode %{} LocalSize 64 1 1\n", i.main));
        for decoration in &e.decorations {
            out.push_str(decoration);
            out.push('\n');
        }
        for ty in &e.types {
            out.push_str(ty);
            out.push('\n');
        }
        out.push_str(&format!(
            "%{} = OpFunction %{} None %{}\n",
            i.main, i.void, i.fn_ty
        ));
        for instruction in &e.body {
            out.push_str(instruction);
            out.push('\n');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::super::program::{DensityProgram, PerlinProgram};
    use std::process::Command;

    /// Assemble and validate an emitted module. The tools are a developer
    /// dependency, so a missing `spirv-as` skips instead of failing; a tool
    /// that runs and rejects the module is always a test failure.
    fn validate(label: &str, module: &str) {
        let dir =
            std::env::temp_dir().join(format!("horizon-spirv-{label}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let spvasm = dir.join("module.spvasm");
        let spv = dir.join("module.spv");
        std::fs::write(&spvasm, module).expect("write spirv text");
        let assembled = match Command::new("spirv-as")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&spvasm)
            .arg("-o")
            .arg(&spv)
            .output()
        {
            Ok(output) => output,
            Err(error) => {
                eprintln!("skipping SPIR-V validation: spirv-as unavailable ({error})");
                let _ = std::fs::remove_dir_all(&dir);
                return;
            }
        };
        assert!(
            assembled.status.success(),
            "spirv-as rejected the module:\n{}\n--- module ---\n{module}",
            String::from_utf8_lossy(&assembled.stderr)
        );
        let validated = Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&spv)
            .output()
            .expect("run spirv-val");
        assert!(
            validated.status.success(),
            "spirv-val rejected the module:\n{}",
            String::from_utf8_lossy(&validated.stderr)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn constant_program_is_valid_spirv() {
        validate(
            "constant",
            &DensityProgram::Constant(0.125).spirv("constant"),
        );
    }

    #[test]
    fn perlin_program_is_valid_spirv() {
        let program = DensityProgram::Perlin(PerlinProgram::from_permutation(&[0u8; 256]));
        validate("perlin", &program.spirv("perlin"));
    }

    #[test]
    fn rung_two_program_is_valid_spirv() {
        use super::super::program::{BinaryKind, UnaryKind};
        let c = DensityProgram::Constant;
        let clamp = DensityProgram::Clamp {
            input: Box::new(c(0.5)),
            min: -1.0,
            max: 1.0,
        };
        let unary = DensityProgram::Unary {
            kind: UnaryKind::Squeeze,
            input: Box::new(DensityProgram::Unary {
                kind: UnaryKind::Sign,
                input: Box::new(c(-0.25)),
            }),
        };
        let binary = DensityProgram::Binary {
            kind: BinaryKind::Maximum,
            left: Box::new(DensityProgram::Binary {
                kind: BinaryKind::Minimum,
                left: Box::new(clamp),
                right: Box::new(c(-0.75)),
            }),
            right: Box::new(DensityProgram::Binary {
                kind: BinaryKind::Divide,
                left: Box::new(unary),
                right: Box::new(c(3.0)),
            }),
        };
        let lerp = DensityProgram::Lerp {
            alpha: Box::new(c(0.375)),
            first: Box::new(binary),
            second: Box::new(DensityProgram::Unary {
                kind: UnaryKind::Reciprocal,
                input: Box::new(c(8.0)),
            }),
        };
        validate("rung-two", &lerp.spirv("rung-two"));
    }

    #[test]
    fn every_vector_extract_uses_a_literal_index() {
        let program = DensityProgram::Perlin(PerlinProgram::from_permutation(&[0u8; 256]));
        let module = program.spirv("perlin");
        for line in module.lines() {
            if let Some(rest) = line.split("OpCompositeExtract").nth(1) {
                let index = rest.split_whitespace().last().expect("extract index");
                assert!(
                    index.parse::<u32>().is_ok(),
                    "OpCompositeExtract needs a literal index, got `{index}`: {line}"
                );
            }
        }
    }
}
