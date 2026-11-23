use super::*;
use crate::density::program::{BinaryKind, DensityProgram, UnaryKind};

/// One reachable graph node that prevents a pointwise export.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExportBlocker {
    pub node_kind: &'static str,
    pub reason: ProgramExportReason,
    /// Shortest distance from the queried root; the root itself is depth 0.
    pub depth: usize,
}

/// How much of one definition graph a pointwise export covers.
#[derive(Clone, Debug, Default)]
pub struct ExportCoverage {
    /// Distinct graph nodes reachable from the root.
    pub reachable: usize,
    /// Every reachable node that prevents export, once per node, ordered by
    /// depth and then reason.
    pub blockers: Vec<ExportBlocker>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProgramExportReason {
    NeedsContext,
    BoundSampler,
    NoiseKind,
    UnsupportedOperation,
    UnknownFunction,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProgramExportError {
    pub node_kind: &'static str,
    pub reason: ProgramExportReason,
}

/// A compiler and cache-ID namespace for one immutable set of density inputs.
/// Share resulting samplers across workers, with a separate SamplerContext per
/// worker. Context cache cells must not be reused across compiler namespaces.
#[derive(Debug)]
pub struct DensityCompiler {
    graph: DensityGraph,
    samplers: HashMap<FunctionId, Arc<dyn DensitySampler>>,
    prepared: HashMap<FunctionId, FunctionId>,
    next_cache: usize,
    context: Option<seeded::SeededContext>,
}
impl DensityCompiler {
    pub fn new(graph: DensityGraph) -> Self {
        Self {
            graph,
            samplers: HashMap::new(),
            prepared: HashMap::new(),
            next_cache: 0,
            context: None,
        }
    }
    pub fn new_seeded(graph: DensityGraph, seed: i64, use_legacy_random: bool) -> Self {
        let mut compiler = Self::new(graph);
        compiler.context = Some(seeded::SeededContext::new(seed, use_legacy_random));
        compiler
    }
    pub fn graph(&self) -> &DensityGraph {
        &self.graph
    }
    pub fn graph_mut(&mut self) -> &mut DensityGraph {
        &mut self.graph
    }
    pub fn cache_count(&self) -> usize {
        self.next_cache
    }
    /// Resolve a registered noise using this compiler's fixed world seed.
    /// Ordinary names reuse their sampler; Nether biome compatibility names
    /// create a fresh sampler from their dedicated legacy seed on each call.
    pub fn get_noise_sampler(&self, noise: NoiseId) -> Result<Arc<dyn Noise>, CompileError> {
        self.context
            .as_ref()
            .ok_or(CompileError::MissingSeededContext)?
            .noise(&self.graph, noise)
    }
    pub fn get_sampler(&mut self, id: FunctionId) -> Result<Arc<dyn DensitySampler>, CompileError> {
        if let Some(sampler) = self.samplers.get(&id) {
            return Ok(sampler.clone());
        }
        let optimized = self.optimize(id)?;
        let sampler = self.compile_raw(optimized)?;
        self.samplers.insert(id, sampler.clone());
        Ok(sampler)
    }

    /// Export the directly representable pointwise subset of this definition
    /// graph. CPU sampler compilation remains independent and unchanged.
    pub fn export_to_program(&self, id: FunctionId) -> Option<DensityProgram> {
        self.try_export_to_program(id).ok()
    }

    /// Detailed form of [`Self::export_to_program`], including the first node
    /// that prevents a pointwise export.
    pub fn try_export_to_program(
        &self,
        id: FunctionId,
    ) -> Result<DensityProgram, ProgramExportError> {
        let entry = self.graph.entry(id).map_err(|_| ProgramExportError {
            node_kind: "unknown function",
            reason: ProgramExportReason::UnknownFunction,
        })?;
        let Definition::Node(node) = &entry.definition else {
            return Err(ProgramExportError {
                node_kind: "bound sampler",
                reason: ProgramExportReason::BoundSampler,
            });
        };
        let unsupported = |node_kind, reason| ProgramExportError { node_kind, reason };
        match node {
            GraphNode::Constant(value) => Ok(DensityProgram::Constant(*value)),
            GraphNode::Reference { input, .. } => self.try_export_to_program(*input),
            GraphNode::Binary { kind, left, right } => {
                let left_program = Box::new(self.try_export_to_program(*left)?);
                let right_program = Box::new(self.try_export_to_program(*right)?);
                // `compile_binary` folds a constant right operand before it
                // builds the sampler. Mirror those two rewrites exactly:
                // `x - c` becomes `x + -c`, and `x / c` becomes
                // `x * (1.0 / c)`. Emitting a raw divide here would differ
                // from the compiled sampler by rounding for some inputs.
                if let Some(value) = self.export_constant(*right) {
                    if self.export_constant(*left).is_none() {
                        match kind {
                            GraphBinary::Subtract => {
                                return Ok(DensityProgram::Binary {
                                    kind: BinaryKind::Add,
                                    left: left_program,
                                    right: Box::new(DensityProgram::Constant(-value)),
                                });
                            }
                            GraphBinary::Divide => {
                                return Ok(DensityProgram::Binary {
                                    kind: BinaryKind::Multiply,
                                    left: left_program,
                                    right: Box::new(DensityProgram::Constant(1.0 / value)),
                                });
                            }
                            _ => {}
                        }
                    }
                }
                let kind = match kind {
                    GraphBinary::Add => BinaryKind::Add,
                    GraphBinary::Subtract => BinaryKind::Subtract,
                    GraphBinary::Multiply => BinaryKind::Multiply,
                    GraphBinary::Divide => BinaryKind::Divide,
                    GraphBinary::Minimum => BinaryKind::Minimum,
                    GraphBinary::Maximum => BinaryKind::Maximum,
                };
                Ok(DensityProgram::Binary {
                    kind,
                    left: left_program,
                    right: right_program,
                })
            }
            GraphNode::Lerp {
                alpha,
                first,
                second,
            } => Ok(DensityProgram::Lerp {
                alpha: Box::new(self.try_export_to_program(*alpha)?),
                first: Box::new(self.try_export_to_program(*first)?),
                second: Box::new(self.try_export_to_program(*second)?),
            }),
            GraphNode::Noise { .. }
            | GraphNode::OffsetNoise { .. }
            | GraphNode::BlendedNoise(_)
            | GraphNode::EndIslands => Err(unsupported(
                graph_node_name(node),
                ProgramExportReason::NoiseKind,
            )),
            // `Cache` is memoization of its child at one position (see
            // `density/cache.rs`: the scalar path stores one packed
            // `BlockPos` and returns it only for the same position), so for a
            // pointwise program it is the identity. `Interpolated` is not:
            // it changes the sampled lattice.
            GraphNode::Cache(input) => self.try_export_to_program(*input),
            GraphNode::WorldField(_)
            | GraphNode::BlendDensity(_)
            | GraphNode::Slice { .. }
            | GraphNode::Interpolated { .. }
            | GraphNode::Surface { .. } => Err(unsupported(
                graph_node_name(node),
                ProgramExportReason::NeedsContext,
            )),
            GraphNode::Clamp { input, min, max } => Ok(DensityProgram::Clamp {
                input: Box::new(self.try_export_to_program(*input)?),
                min: *min,
                max: *max,
            }),
            GraphNode::Unary { kind, input } => {
                let kind = match kind {
                    GraphUnary::Abs => UnaryKind::Abs,
                    GraphUnary::Negate => UnaryKind::Negate,
                    GraphUnary::Square => UnaryKind::Square,
                    GraphUnary::Cube => UnaryKind::Cube,
                    GraphUnary::HalfNegative => UnaryKind::HalfNegative,
                    GraphUnary::QuarterNegative => UnaryKind::QuarterNegative,
                    GraphUnary::Reciprocal => UnaryKind::Reciprocal,
                    GraphUnary::Sign => UnaryKind::Sign,
                    GraphUnary::Squeeze => UnaryKind::Squeeze,
                    // `f64` libm behind a `f32` result: not bit-reproducible
                    // on a GPU, so these stay off the exact tier.
                    GraphUnary::Sqrt | GraphUnary::Log => {
                        return Err(unsupported(
                            "Unary",
                            ProgramExportReason::UnsupportedOperation,
                        ));
                    }
                };
                Ok(DensityProgram::Unary {
                    kind,
                    input: Box::new(self.try_export_to_program(*input)?),
                })
            }
            GraphNode::Spline(_) => Err(unsupported(
                "Spline",
                ProgramExportReason::UnsupportedOperation,
            )),
            GraphNode::Gradient { .. }
            | GraphNode::Round { .. }
            | GraphNode::Power { .. }
            | GraphNode::Choice { .. }
            | GraphNode::IntervalSelect { .. }
            | GraphNode::DistanceToPoint { .. } => Err(unsupported(
                graph_node_name(node),
                ProgramExportReason::UnsupportedOperation,
            )),
        }
    }

    /// Return each reachable node that prevents export, once per graph node.
    /// This gives callers a complete coverage report instead of only the first
    /// error returned by [`Self::try_export_to_program`].
    pub fn export_blockers(&self, id: FunctionId) -> Vec<ProgramExportError> {
        fn visit(
            compiler: &DensityCompiler,
            id: FunctionId,
            seen: &mut std::collections::HashSet<FunctionId>,
            blockers: &mut Vec<ProgramExportError>,
        ) {
            if !seen.insert(id) {
                return;
            }
            let Ok(entry) = compiler.graph.entry(id) else {
                blockers.push(ProgramExportError {
                    node_kind: "unknown function",
                    reason: ProgramExportReason::UnknownFunction,
                });
                return;
            };
            let Definition::Node(mut node) = entry.definition.clone() else {
                blockers.push(ProgramExportError {
                    node_kind: "bound sampler",
                    reason: ProgramExportReason::BoundSampler,
                });
                return;
            };
            if let Some(failure) = export_node_failure(&node) {
                blockers.push(failure);
            }
            for child in node.children_mut() {
                visit(compiler, *child, seen, blockers);
            }
        }

        let mut blockers = Vec::new();
        visit(
            self,
            id,
            &mut std::collections::HashSet::new(),
            &mut blockers,
        );
        blockers
    }

    /// Coverage report for one root: the number of distinct reachable graph
    /// nodes and every node that prevents a pointwise export, each with its
    /// shortest depth from the root.
    ///
    /// [`Self::export_blockers`] stops at the first failure along a path and
    /// reports no depth. This walks the whole graph once, which is what an
    /// export-coverage measurement of a large decoded router needs.
    pub fn export_coverage(&self, id: FunctionId) -> ExportCoverage {
        let mut depths: HashMap<FunctionId, usize> = HashMap::new();
        let mut pending = vec![(id, 0usize)];
        while let Some((node, depth)) = pending.pop() {
            if depths.get(&node).is_some_and(|known| *known <= depth) {
                continue;
            }
            depths.insert(node, depth);
            let Ok(entry) = self.graph.entry(node) else {
                continue;
            };
            let Definition::Node(mut definition) = entry.definition.clone() else {
                continue;
            };
            for child in definition.children_mut() {
                pending.push((*child, depth + 1));
            }
        }
        let mut blockers = Vec::new();
        for (node, depth) in &depths {
            let Ok(entry) = self.graph.entry(*node) else {
                continue;
            };
            let blocker = match &entry.definition {
                Definition::Node(definition) => {
                    export_node_failure(definition).map(|failure| ExportBlocker {
                        node_kind: failure.node_kind,
                        reason: failure.reason,
                        depth: *depth,
                    })
                }
                Definition::Bound { .. } => Some(ExportBlocker {
                    node_kind: "bound sampler",
                    reason: ProgramExportReason::BoundSampler,
                    depth: *depth,
                }),
            };
            blockers.extend(blocker);
        }
        blockers.sort_by_key(|blocker| (blocker.depth, blocker.reason as u8, blocker.node_kind));
        ExportCoverage {
            reachable: depths.len(),
            blockers,
        }
    }
    fn rewrite_children(
        &mut self,
        id: FunctionId,
        mut rewrite: impl FnMut(&mut Self, FunctionId) -> Result<FunctionId, CompileError>,
    ) -> Result<FunctionId, CompileError> {
        let Definition::Node(mut node) = self.graph.entry(id)?.definition.clone() else {
            return Ok(id);
        };
        if let GraphNode::Reference { input, .. } = node {
            let rewritten = rewrite(self, input)?;
            return Ok(if rewritten == input { id } else { rewritten });
        }
        for child in node.children_mut() {
            *child = rewrite(self, *child)?;
        }
        self.graph.insert(node)
    }
    fn optimize(&mut self, id: FunctionId) -> Result<FunctionId, CompileError> {
        let prepared = self.prepare(id)?;
        self.slice_uniform_axes(prepared, 7)
    }
    fn prepare(&mut self, mut id: FunctionId) -> Result<FunctionId, CompileError> {
        // Vanilla INLINE_REFERENCE unwraps one holder before detecting caches.
        // Nested holders still follow their own rewriteChildren identity rule.
        if let Definition::Node(GraphNode::Reference { input, .. }) =
            self.graph.entry(id)?.definition
        {
            id = input;
        }
        if let Definition::Node(GraphNode::Cache(input)) = self.graph.entry(id)?.definition.clone()
        {
            if let Some(prepared) = self.prepared.get(&input) {
                return Ok(*prepared);
            }
            let cache_id = self.next_cache;
            self.next_cache += 1;
            let optimized = self.optimize(input)?;
            let sampler = self.compile_raw(optimized)?;
            let sampler = Arc::new(CachingDensitySampler::new(cache_id, sampler)?);
            let range = self.graph.range(optimized)?;
            let axes = self.graph.domain_axes(optimized)?;
            let prepared = self.graph.bind(sampler, range, axes)?;
            self.prepared.insert(input, prepared);
            return Ok(prepared);
        }
        self.rewrite_children(id, Self::prepare)
    }
    fn slice_uniform_axes(
        &mut self,
        id: FunctionId,
        parent_axes: u8,
    ) -> Result<FunctionId, CompileError> {
        if matches!(
            self.graph.entry(id)?.definition,
            Definition::Node(GraphNode::Constant(_) | GraphNode::Gradient { .. })
        ) {
            return Ok(id);
        }
        let axes = self.graph.domain_axes(id)?;
        let mut result = self.rewrite_children(id, |compiler, child| {
            compiler.slice_uniform_axes(child, axes)
        })?;
        if axes == parent_axes {
            return Ok(result);
        }
        let mut existing = 0;
        let mut cursor = result;
        while let Definition::Node(GraphNode::Slice { axis, input, .. }) =
            self.graph.entry(cursor)?.definition
        {
            existing |= 1 << axis as u8;
            cursor = input;
        }
        let remove = parent_axes & !axes & !existing;
        // X then Z places these slices adjacent for the XZ sampler specialization.
        for axis in [Axis::X, Axis::Z, Axis::Y] {
            if remove & (1 << axis as u8) != 0 {
                result = self.graph.insert(GraphNode::Slice {
                    axis,
                    coordinate: 0,
                    input: result,
                })?;
            }
        }
        Ok(result)
    }
    fn constant(&self, id: FunctionId) -> Option<f32> {
        match self.graph.entries[id.0].definition {
            Definition::Node(GraphNode::Constant(value)) => Some(value),
            _ => None,
        }
    }
    /// Constant lookup for the exporter. `prepare` unwraps a `Reference` before
    /// `compile_binary` runs, so the exporter must look through it too or it
    /// would take a different branch than the compiled sampler.
    fn export_constant(&self, id: FunctionId) -> Option<f32> {
        match self.graph.entries[id.0].definition {
            Definition::Node(GraphNode::Constant(value)) => Some(value),
            Definition::Node(GraphNode::Reference { input, .. }) => self.export_constant(input),
            _ => None,
        }
    }
    /// Compile exactly the supplied definition, without rewriting. Cache nodes
    /// require get_sampler's preparation pass and are rejected here.
    pub fn compile_raw(&self, id: FunctionId) -> Result<Arc<dyn DensitySampler>, CompileError> {
        let node = match &self.graph.entry(id)?.definition {
            Definition::Bound { sampler, .. } => return Ok(sampler.clone()),
            Definition::Node(node) => node,
        };
        let sampler: Arc<dyn DensitySampler> = match *node {
            GraphNode::Constant(value) => Arc::new(ConstantSampler(value)),
            GraphNode::Reference { input, .. } => self.compile_raw(input)?,
            GraphNode::WorldField(field) => Arc::new(ContextBoundSampler::new(field)),
            GraphNode::BlendDensity(input) => Arc::new(BlendDensitySampler {
                input: self.compile_raw(input)?,
            }),
            GraphNode::EndIslands => self
                .context
                .as_ref()
                .ok_or(CompileError::MissingSeededContext)?
                .end_islands(),
            GraphNode::Noise {
                noise,
                xz_scale,
                y_scale,
                shift_x,
                shift_y,
                shift_z,
            } => {
                let noise = self
                    .context
                    .as_ref()
                    .ok_or(CompileError::MissingSeededContext)?
                    .noise(&self.graph, noise)?;
                let zero = |id: FunctionId| matches!(self.graph.entries[id.0].definition, Definition::Node(GraphNode::Constant(value)) if value.to_bits() == 0);
                if zero(shift_x) && zero(shift_y) && zero(shift_z) {
                    Arc::new(NoiseSampler {
                        noise,
                        xz_scale,
                        y_scale,
                    })
                } else {
                    // Compilation order differs from the XYZ sampler's sampling
                    // order: vanilla compiles X, then Z, then optional Y.
                    let x = self.compile_raw(shift_x)?;
                    let z = self.compile_raw(shift_z)?;
                    let y = if zero(shift_y) {
                        None
                    } else {
                        Some(self.compile_raw(shift_y)?)
                    };
                    Arc::new(ShiftedNoiseSampler {
                        source: NoiseSampler {
                            noise,
                            xz_scale,
                            y_scale,
                        },
                        shift_x: x,
                        shift_y: y,
                        shift_z: z,
                    })
                }
            }
            GraphNode::OffsetNoise { noise, kind } => compile_shift_noise(
                self.context
                    .as_ref()
                    .ok_or(CompileError::MissingSeededContext)?
                    .noise(&self.graph, noise)?,
                kind,
            ),
            GraphNode::BlendedNoise(definition) => self
                .context
                .as_ref()
                .ok_or(CompileError::MissingSeededContext)?
                .blended(definition)?,
            GraphNode::Spline(ref spline) => {
                let mut ids = HashMap::new();
                let mut coordinates = Vec::new();
                let compiled = spline.try_map_coordinates(&mut |id| {
                    if let Some(index) = ids.get(&id) {
                        return Ok::<usize, CompileError>(*index);
                    }
                    let sampler = self.compile_raw(id)?;
                    // Vanilla validates each unique coordinate range immediately
                    // after compiling it, even if sampling never selects it.
                    self.graph.range(id)?;
                    let index = coordinates.len();
                    coordinates.push(sampler);
                    ids.insert(id, index);
                    Ok(index)
                })?;
                Arc::new(SplineSampler::new(&compiled, &coordinates)?)
            }
            GraphNode::DistanceToPoint { point, metric } => {
                Arc::new(DistanceToPointSampler { point, metric })
            }
            GraphNode::IntervalSelect {
                input,
                ref thresholds,
                ref branches,
            } => {
                let input = self.compile_raw(input)?;
                let branches = branches
                    .iter()
                    .map(|id| self.compile_raw(*id))
                    .collect::<Result<Vec<_>, _>>()?;
                Arc::new(
                    IntervalSelectSampler::new(input, thresholds.to_vec(), branches)
                        .map_err(|_| CompileError::InvalidIntervals)?,
                )
            }
            GraphNode::Gradient {
                axis,
                tiling,
                from,
                to,
                low,
                high,
            } => Arc::new(GradientSampler::new(axis, tiling, from, to, low, high)),
            GraphNode::Unary { kind, input } => Arc::new(UnarySampler {
                input: self.compile_raw(input)?,
                operation: kind.operation(),
            }),
            GraphNode::Round {
                mode,
                input,
                multiple,
            } => {
                let input = self.compile_raw(input)?;
                let multiple = if self.constant(multiple) == Some(1.0) {
                    None
                } else {
                    Some(self.compile_raw(multiple)?)
                };
                Arc::new(RoundSampler {
                    mode,
                    input,
                    multiple,
                })
            }
            GraphNode::Power { base, exponent } => {
                // Both children compile before literal selection, even when one
                // compiled child will be discarded by the specialization.
                let compiled_base = self.compile_raw(base)?;
                let compiled_exponent = self.compile_raw(exponent)?;
                if let Some(value) = self.constant(base) {
                    Arc::new(UnarySampler {
                        input: compiled_exponent,
                        operation: UnaryOp::PowerBase(value as f64),
                    })
                } else if let Some(value) = self.constant(exponent) {
                    compile_constant_exponent(compiled_base, value)
                } else {
                    Arc::new(PowerSampler {
                        base: compiled_base,
                        exponent: compiled_exponent,
                    })
                }
            }
            GraphNode::Binary { kind, left, right } => {
                return self.compile_binary(kind, left, right);
            }
            GraphNode::Clamp { input, min, max } => Arc::new(UnarySampler {
                input: self.compile_raw(input)?,
                operation: UnaryOp::Clamp { min, max },
            }),
            GraphNode::Choice {
                input,
                min,
                max,
                inside,
                outside,
            } => {
                let input = self.compile_raw(input)?;
                if let (Some(inside), Some(outside)) =
                    (self.constant(inside), self.constant(outside))
                {
                    Arc::new(UnarySampler {
                        input,
                        operation: UnaryOp::ConstantRangeChoice {
                            min_inclusive: min,
                            max_exclusive: max,
                            inside,
                            outside,
                        },
                    })
                } else {
                    Arc::new(RangeChoiceSampler {
                        input,
                        min_inclusive: min,
                        max_exclusive: max,
                        when_in_range: self.compile_raw(inside)?,
                        when_out_of_range: self.compile_raw(outside)?,
                    })
                }
            }
            GraphNode::Lerp {
                alpha,
                first,
                second,
            } => {
                let alpha = self.compile_raw(alpha)?;
                let first_sampler = self.compile_raw(first)?;
                let second_sampler = self.compile_raw(second)?;
                let (first, second) = if let Some(value) = self.constant(first) {
                    (
                        DensityValue::Constant(value),
                        DensityValue::Sampler(second_sampler),
                    )
                } else if let Some(value) = self.constant(second) {
                    (
                        DensityValue::Sampler(first_sampler),
                        DensityValue::Constant(value),
                    )
                } else {
                    (
                        DensityValue::Sampler(first_sampler),
                        DensityValue::Sampler(second_sampler),
                    )
                };
                Arc::new(LerpSampler {
                    alpha,
                    first,
                    second,
                })
            }
            GraphNode::Slice {
                axis,
                coordinate,
                input,
            } => {
                let mut axes = match axis {
                    Axis::X => SliceAxes::X(coordinate),
                    Axis::Y => SliceAxes::Y(coordinate),
                    Axis::Z => SliceAxes::Z(coordinate),
                };
                let mut child = input;
                if let Definition::Node(GraphNode::Slice {
                    axis: inner_axis,
                    coordinate: inner_coordinate,
                    input: inner,
                }) = self.graph.entry(input)?.definition
                {
                    if axis == Axis::X && inner_axis == Axis::Z {
                        axes = SliceAxes::Xz {
                            x: coordinate,
                            z: inner_coordinate,
                        };
                        child = inner;
                    } else if axis == Axis::Z && inner_axis == Axis::X {
                        axes = SliceAxes::Xz {
                            x: inner_coordinate,
                            z: coordinate,
                        };
                        child = inner;
                    }
                }
                Arc::new(SliceSampler {
                    axes,
                    input: self.compile_raw(child)?,
                })
            }
            GraphNode::Cache(_) => return Err(CompileError::UnpreparedCache),
            GraphNode::Interpolated { input, xz, y } => {
                Arc::new(InterpolatedSampler::new(self.compile_raw(input)?, xz, y)?)
            }
            GraphNode::Surface {
                density,
                upper,
                lower,
                height,
            } => Arc::new(compile_surface_search(
                self.compile_raw(density)?,
                self.compile_raw(upper)?,
                lower,
                height,
            )?),
        };
        Ok(sampler)
    }
    fn compile_binary(
        &self,
        kind: GraphBinary,
        left_id: FunctionId,
        right_id: FunctionId,
    ) -> Result<Arc<dyn DensitySampler>, CompileError> {
        let left = self.compile_raw(left_id)?;
        let right = self.compile_raw(right_id)?;
        let unary = |input, operation| {
            Arc::new(UnarySampler { input, operation }) as Arc<dyn DensitySampler>
        };
        let bounds = if matches!(kind, GraphBinary::Minimum | GraphBinary::Maximum) {
            let a = self.graph.range(left_id)?;
            let b = self.graph.range(right_id)?;
            if (kind == GraphBinary::Minimum && a.max() < b.min())
                || (kind == GraphBinary::Maximum && a.min() > b.max())
            {
                return Ok(left);
            }
            if (kind == GraphBinary::Minimum && b.max() < a.min())
                || (kind == GraphBinary::Maximum && b.min() > a.max())
            {
                return Ok(right);
            }
            Some(b)
        } else {
            None
        };
        if let Some(value) = self.constant(left_id) {
            return Ok(unary(
                right,
                match kind {
                    GraphBinary::Add => UnaryOp::Add(value),
                    GraphBinary::Subtract => UnaryOp::SubtractFrom(value),
                    GraphBinary::Multiply => UnaryOp::Multiply(value),
                    GraphBinary::Divide => UnaryOp::DivideInto(value),
                    GraphBinary::Minimum => UnaryOp::Minimum(value),
                    GraphBinary::Maximum => UnaryOp::Maximum(value),
                },
            ));
        }
        if let Some(value) = self.constant(right_id) {
            return Ok(unary(
                left,
                match kind {
                    GraphBinary::Add => UnaryOp::Add(value),
                    GraphBinary::Subtract => UnaryOp::Add(-value),
                    GraphBinary::Multiply => UnaryOp::Multiply(value),
                    GraphBinary::Divide => UnaryOp::Multiply(1.0 / value),
                    GraphBinary::Minimum => UnaryOp::Minimum(value),
                    GraphBinary::Maximum => UnaryOp::Maximum(value),
                },
            ));
        }
        let operation = match kind {
            GraphBinary::Add => BinaryOp::Add,
            GraphBinary::Subtract => BinaryOp::Subtract,
            GraphBinary::Multiply => BinaryOp::Multiply,
            GraphBinary::Divide => BinaryOp::Divide,
            GraphBinary::Minimum => BinaryOp::Minimum {
                right_min: bounds.unwrap().min(),
            },
            GraphBinary::Maximum => BinaryOp::Maximum {
                right_max: bounds.unwrap().max(),
            },
        };
        Ok(Arc::new(BinarySampler {
            left,
            right,
            operation,
        }))
    }
}

fn graph_node_name(node: &GraphNode) -> &'static str {
    match node {
        GraphNode::Constant(_) => "Constant",
        GraphNode::Spline(_) => "Spline",
        GraphNode::Noise { .. } => "Noise",
        GraphNode::OffsetNoise { .. } => "OffsetNoise",
        GraphNode::BlendedNoise(_) => "BlendedNoise",
        GraphNode::EndIslands => "EndIslands",
        GraphNode::WorldField(_) => "WorldField",
        GraphNode::BlendDensity(_) => "BlendDensity",
        GraphNode::Reference { .. } => "Reference",
        GraphNode::Gradient { .. } => "Gradient",
        GraphNode::Unary { .. } => "Unary",
        GraphNode::Binary { .. } => "Binary",
        GraphNode::Round { .. } => "Round",
        GraphNode::Power { .. } => "Power",
        GraphNode::Clamp { .. } => "Clamp",
        GraphNode::Choice { .. } => "Choice",
        GraphNode::Lerp { .. } => "Lerp",
        GraphNode::Slice { .. } => "Slice",
        GraphNode::Cache(_) => "Cache",
        GraphNode::Interpolated { .. } => "Interpolated",
        GraphNode::Surface { .. } => "Surface",
        GraphNode::IntervalSelect { .. } => "IntervalSelect",
        GraphNode::DistanceToPoint { .. } => "DistanceToPoint",
    }
}

fn export_node_failure(node: &GraphNode) -> Option<ProgramExportError> {
    let (reason, supported) = match node {
        GraphNode::Constant(_)
        | GraphNode::Reference { .. }
        | GraphNode::Lerp { .. }
        | GraphNode::Cache(_) => (ProgramExportReason::UnsupportedOperation, true),
        GraphNode::Binary { .. } | GraphNode::Clamp { .. } => {
            (ProgramExportReason::UnsupportedOperation, true)
        }
        GraphNode::Unary { kind, .. } => (
            ProgramExportReason::UnsupportedOperation,
            !matches!(kind, GraphUnary::Sqrt | GraphUnary::Log),
        ),
        GraphNode::Noise { .. }
        | GraphNode::OffsetNoise { .. }
        | GraphNode::BlendedNoise(_)
        | GraphNode::EndIslands => (ProgramExportReason::NoiseKind, false),
        GraphNode::WorldField(_)
        | GraphNode::BlendDensity(_)
        | GraphNode::Slice { .. }
        | GraphNode::Interpolated { .. }
        | GraphNode::Surface { .. } => (ProgramExportReason::NeedsContext, false),
        GraphNode::Spline(_)
        | GraphNode::Gradient { .. }
        | GraphNode::Round { .. }
        | GraphNode::Power { .. }
        | GraphNode::Choice { .. }
        | GraphNode::IntervalSelect { .. }
        | GraphNode::DistanceToPoint { .. } => (ProgramExportReason::UnsupportedOperation, false),
    };
    (!supported).then(|| ProgramExportError {
        node_kind: graph_node_name(node),
        reason,
    })
}
