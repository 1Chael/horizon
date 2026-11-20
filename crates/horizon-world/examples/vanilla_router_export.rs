//! Vanilla router export coverage: can the default noise settings' density
//! graphs be lowered to the pointwise `DensityProgram` IR, and if not, which
//! graph nodes block it and how far from each root they sit?
//!
//! vanilla_router_export <worldgen bundle dir> [--settings NAME] [--summary]
//!
//! The bundle is the prepared `worldgen/` directory (the `GeneratorData`
//! layout): `noises.json`, `densities.json` and `settings.json`, each a map
//! from registry identifier to original JSON text. Only decoding and export
//! are exercised; nothing is sampled and no world is generated.
use horizon_core::density::graph::{
    DensityCompiler, ExportCoverage, ProgramExportError, ProgramExportReason,
};
use horizon_world::worldgen::density::DensityRegistry;
use horizon_world::worldgen::router::{NoiseRouter, RouterField};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

type Error = Box<dyn std::error::Error + Send + Sync>;

fn reason_name(reason: ProgramExportReason) -> &'static str {
    match reason {
        ProgramExportReason::NeedsContext => "NeedsContext",
        ProgramExportReason::BoundSampler => "BoundSampler",
        ProgramExportReason::NoiseKind => "NoiseKind",
        ProgramExportReason::UnsupportedOperation => "UnsupportedOperation",
        ProgramExportReason::UnknownFunction => "UnknownFunction",
    }
}

fn family(reason: ProgramExportReason) -> &'static str {
    reason_name(reason)
}

fn print_coverage(coverage: &ExportCoverage) {
    let mut by_kind: BTreeMap<(&'static str, &'static str), (usize, usize, usize)> =
        BTreeMap::new();
    for blocker in &coverage.blockers {
        let entry = by_kind
            .entry((family(blocker.reason), blocker.node_kind))
            .or_insert((0, usize::MAX, 0));
        entry.0 += 1;
        entry.1 = entry.1.min(blocker.depth);
        entry.2 = entry.2.max(blocker.depth);
    }
    let mut families: BTreeMap<&'static str, usize> = BTreeMap::new();
    for blocker in &coverage.blockers {
        *families.entry(family(blocker.reason)).or_default() += 1;
    }
    println!(
        "  blockers {} / reachable {}; families: {}",
        coverage.blockers.len(),
        coverage.reachable,
        families
            .iter()
            .map(|(name, count)| format!("{name} {count}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for ((family, kind), (count, min, max)) in by_kind {
        println!("    {family:<19} {kind:<16} {count:>4}  depth {min}..={max}");
    }
}

fn print_root(
    compiler: &DensityCompiler,
    router: NoiseRouter,
    field: RouterField,
    summary: bool,
) -> (bool, BTreeSet<&'static str>) {
    let id = router.function(field);
    let export = compiler.try_export_to_program(id);
    let coverage = compiler.export_coverage(id);
    let first = match &export {
        Ok(_) => "-".to_owned(),
        Err(ProgramExportError { node_kind, reason }) => {
            format!("{node_kind} ({})", reason_name(*reason))
        }
    };
    let families: BTreeSet<&'static str> = coverage
        .blockers
        .iter()
        .map(|blocker| family(blocker.reason))
        .collect();
    println!(
        "{:<20} {:<5} reachable {:<6} supported {:<6} blockers {:<5} first {}",
        field.name(),
        if export.is_ok() { "yes" } else { "no" },
        coverage.reachable,
        coverage.reachable - coverage.blockers.len(),
        coverage.blockers.len(),
        first
    );
    if !families.is_empty() {
        println!(
            "  needs: {}",
            families.iter().copied().collect::<Vec<_>>().join(", ")
        );
    }
    if !summary && !coverage.blockers.is_empty() {
        print_coverage(&coverage);
    }
    (export.is_ok(), families)
}

fn main() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(
        args.next()
            .ok_or("usage: vanilla_router_export <worldgen>")?,
    );
    let mut only: Option<String> = None;
    let mut summary = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--settings" => only = Some(args.next().ok_or("--settings NAME")?),
            "--summary" => summary = true,
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
    println!(
        "bundle {}: noises {}, densities {}, settings {}",
        dir.display(),
        noises.len(),
        densities.len(),
        settings.len()
    );
    for (name, text) in &settings {
        if only.as_deref().is_some_and(|only| only != name) {
            continue;
        }
        println!("\n== {name} ==");
        let mut registry = DensityRegistry::from_json(&noises, &densities)?;
        let value: serde_json::Value = serde_json::from_str(text)?;
        let router = NoiseRouter::from_json(&value["noise_router"].to_string(), &mut registry)?;
        let compiler = DensityCompiler::new(registry.into_graph());
        let mut fields: BTreeMap<&'static str, (bool, BTreeSet<&'static str>)> = BTreeMap::new();
        for field in RouterField::ALL {
            fields.insert(field.name(), print_root(&compiler, router, field, summary));
        }
        let exported = fields.values().filter(|(ok, _)| *ok).count();
        println!("exported {exported} of {} router fields", fields.len());
        // Which family sets would unblock which fields: a field exports once
        // every family in its set is supported. Report the minimal sets.
        let mut by_set: BTreeMap<Vec<&'static str>, Vec<&'static str>> = BTreeMap::new();
        for (name, (ok, families)) in &fields {
            if !ok {
                by_set
                    .entry(families.iter().copied().collect())
                    .or_default()
                    .push(name);
            }
        }
        for (families, names) in by_set {
            println!("  after {}: {}", families.join(" + "), names.join(", "));
        }
    }
    Ok(())
}
