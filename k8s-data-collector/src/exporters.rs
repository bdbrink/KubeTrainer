use anyhow::{Context, Result};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use crate::models::TrainingExample;

pub fn export_training_data(
    data: &[TrainingExample],
    output_path: &Path,
    format: &crate::OutputFormat,
) -> Result<()> {
    match format {
        crate::OutputFormat::Jsonl => export_jsonl(data, output_path),
        crate::OutputFormat::Json => export_json(data, output_path),
        crate::OutputFormat::Csv => export_csv(data, output_path),
    }
}

fn export_jsonl(data: &[TrainingExample], output_path: &Path) -> Result<()> {
    let path = output_path.with_extension("jsonl");
    let file = File::create(&path)
        .context(format!("Failed to create file: {:?}", path))?;
    let mut writer = BufWriter::new(file);
    
    for example in data {
        let json = serde_json::to_string(example)?;
        writeln!(writer, "{}", json)?;
    }
    
    writer.flush()?;
    tracing::info!("📝 Wrote {} examples to {:?}", data.len(), path);
    Ok(())
}

fn export_json(data: &[TrainingExample], output_path: &Path) -> Result<()> {
    let path = output_path.with_extension("json");
    let file = File::create(&path)
        .context(format!("Failed to create file: {:?}", path))?;
    
    serde_json::to_writer_pretty(file, &data)?;
    
    tracing::info!("📝 Wrote {} examples to {:?}", data.len(), path);
    Ok(())
}

fn export_csv(data: &[TrainingExample], output_path: &Path) -> Result<()> {
    let path = output_path.with_extension("csv");
    let file = File::create(&path)
        .context(format!("Failed to create file: {:?}", path))?;
    
    let mut writer = csv::Writer::from_writer(file);
    
    // Write header
    writer.write_record(&[
        "id",
        "resource_type",
        "input",
        "output",
        "severity",
        "namespace",
        "timestamp",
    ])?;
    
    // Write data
    for example in data {
        writer.write_record(&[
            &example.id,
            &example.resource_type,
            &example.input,
            &example.output,
            &format!("{:?}", example.metadata.severity),
            example.metadata.namespace.as_deref().unwrap_or(""),
            &example.timestamp.to_rfc3339(),
        ])?;
    }
    
    writer.flush()?;
    tracing::info!("📝 Wrote {} examples to {:?}", data.len(), path);
    Ok(())
}