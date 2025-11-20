use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use kube::{
    api::{Api, ListParams},
    Client,
};
use k8s_openapi::api::{
    apps::v1::{Deployment, StatefulSet, DaemonSet},
    core::v1::{Pod, Event, Node, PersistentVolumeClaim, PersistentVolume, Service},
    batch::v1::Job,
    storage::v1::StorageClass,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::info;

mod collectors;
mod exporters;
mod models;

use collectors::*;
use exporters::*;
use models::*;

#[derive(Parser)]
#[command(author, version, about = "K8s Data Collector for SRE AI Training")]
struct Cli {
    /// Output directory for collected data
    #[arg(short, long, default_value = "./training_data")]
    output_dir: PathBuf,

    /// Kubernetes namespace (empty = all namespaces)
    #[arg(short, long)]
    namespace: Option<String>,

    /// Output format
    #[arg(short, long, value_enum, default_value = "jsonl")]
    format: OutputFormat,

    /// Kubeconfig path (uses default if not specified)
    #[arg(short, long)]
    kubeconfig: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Collect all available data
    All {
        /// Include historical events
        #[arg(long)]
        include_events: bool,
    },
    
    /// Collect pod information and issues
    Pods {
        /// Only failing/problematic pods
        #[arg(long)]
        problems_only: bool,
    },
    
    /// Collect deployment configurations
    Deployments,
    
    /// Collect events (warnings, errors)
    Events {
        /// Only warning/error events
        #[arg(long)]
        problems_only: bool,
        
        /// Time window in hours
        #[arg(long, default_value = "24")]
        hours: u64,
    },
    
    /// Collect node health and capacity
    Nodes,
    
    /// Collect StatefulSets
    StatefulSets,
    
    /// Collect DaemonSets
    DaemonSets,
    
    /// Collect Jobs
    Jobs,
    
    /// Collect Services
    Services,
    
    /// Collect PersistentVolumeClaims
    Pvcs,
    
    /// Collect PersistentVolumes
    Pvs,
    
    /// Collect StorageClasses
    StorageClasses,
    
    /// Generate synthetic SRE scenarios
    Synthetic {
        /// Number of scenarios to generate
        #[arg(short, long, default_value = "100")]
        count: usize,
    },
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum OutputFormat {
    /// JSON Lines (one JSON object per line)
    Jsonl,
    /// CSV format
    Csv,
    /// Python pickle-compatible JSON
    Json,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("k8s_data_collector=info")
        .init();

    let cli = Cli::parse();

    // Create output directory
    std::fs::create_dir_all(&cli.output_dir)
        .context("Failed to create output directory")?;

    info!("🚀 K8s Data Collector starting...");
    info!("📁 Output directory: {:?}", cli.output_dir);

    // Check if we need a K8s client (synthetic mode doesn't need one)
    let needs_k8s_client = !matches!(cli.command, Commands::Synthetic { .. });

    // Initialize Kubernetes client only if needed
    let client = if needs_k8s_client {
        if let Some(ref _kubeconfig_path) = cli.kubeconfig {
            let config = kube::Config::from_kubeconfig(&kube::config::KubeConfigOptions {
                context: None,
                cluster: None,
                user: None,
            })
            .await?;
            Some(Client::try_from(config)?)
        } else {
            Some(Client::try_default().await?)
        }
    } else {
        None
    };

    if let Some(ref client) = client {
        info!("✅ Connected to Kubernetes cluster");
    } else {
        info!("ℹ️  Running in offline mode (no K8s connection needed)");
    }

    // Execute command
    match cli.command {
        Commands::All { include_events } => {
            let client = client.as_ref().expect("K8s client required for 'all' command");
            collect_all_data(client, &cli, include_events).await?;
        }
        Commands::Pods { problems_only } => {
            let client = client.as_ref().expect("K8s client required for 'pods' command");
            collect_pods(client, &cli, problems_only).await?;
        }
        Commands::Deployments => {
            let client = client.as_ref().expect("K8s client required for 'deployments' command");
            collect_deployments(client, &cli).await?;
        }
        Commands::Events { problems_only, hours } => {
            let client = client.as_ref().expect("K8s client required for 'events' command");
            collect_events(client, &cli, problems_only, hours).await?;
        }
        Commands::Nodes => {
            let client = client.as_ref().expect("K8s client required for 'nodes' command");
            collect_nodes(client, &cli).await?;
        }
        Commands::StatefulSets => {
            let client = client.as_ref().expect("K8s client required for 'statefulsets' command");
            collect_statefulsets(client, &cli).await?;
        }
        Commands::DaemonSets => {
            let client = client.as_ref().expect("K8s client required for 'daemonsets' command");
            collect_daemonsets(client, &cli).await?;
        }
        Commands::Jobs => {
            let client = client.as_ref().expect("K8s client required for 'jobs' command");
            collect_jobs(client, &cli).await?;
        }
        Commands::Services => {
            let client = client.as_ref().expect("K8s client required for 'services' command");
            collect_services(client, &cli).await?;
        }
        Commands::Pvcs => {
            let client = client.as_ref().expect("K8s client required for 'pvcs' command");
            collect_pvcs(client, &cli).await?;
        }
        Commands::Pvs => {
            let client = client.as_ref().expect("K8s client required for 'pvs' command");
            collect_pvs(client, &cli).await?;
        }
        Commands::StorageClasses => {
            let client = client.as_ref().expect("K8s client required for 'storageclasses' command");
            collect_storageclasses(client, &cli).await?;
        }
        Commands::Synthetic { count } => {
            // Synthetic mode doesn't need K8s client
            generate_synthetic_data(&cli, count).await?;
        }
    }

    info!("🎉 Data collection complete!");
    
    Ok(())
}

async fn collect_all_data(client: &Client, cli: &Cli, include_events: bool) -> Result<()> {
    info!("📊 Collecting all cluster data...");
    
    collect_pods(client, cli, false).await?;
    collect_deployments(client, cli).await?;
    collect_statefulsets(client, cli).await?;
    collect_daemonsets(client, cli).await?;
    collect_jobs(client, cli).await?;
    collect_services(client, cli).await?;
    collect_pvcs(client, cli).await?;
    collect_pvs(client, cli).await?;
    collect_storageclasses(client, cli).await?;
    collect_nodes(client, cli).await?;
    
    if include_events {
        collect_events(client, cli, false, 24).await?;
    }
    
    Ok(())
}

async fn collect_pods(client: &Client, cli: &Cli, problems_only: bool) -> Result<()> {
    info!("🔍 Collecting pod data...");
    
    let pods: Api<Pod> = if let Some(ns) = &cli.namespace {
        Api::namespaced(client.clone(), ns)
    } else {
        Api::all(client.clone())
    };

    let lp = ListParams::default();
    let pod_list = pods.list(&lp).await?;
    
    let mut training_data = Vec::new();
    
    for pod in pod_list {
        let pod_data = PodTrainingData::from_pod(pod);
        
        if problems_only && !pod_data.has_problems() {
            continue;
        }
        
        training_data.push(TrainingExample::from_pod(pod_data));
    }
    
    let output_path = cli.output_dir.join("pods_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Collected {} pod examples", training_data.len());
    
    Ok(())
}

async fn collect_deployments(client: &Client, cli: &Cli) -> Result<()> {
    info!("🔍 Collecting deployment data...");
    
    let deployments: Api<Deployment> = if let Some(ns) = &cli.namespace {
        Api::namespaced(client.clone(), ns)
    } else {
        Api::all(client.clone())
    };

    let lp = ListParams::default();
    let dep_list = deployments.list(&lp).await?;
    
    let mut training_data = Vec::new();
    
    for deployment in dep_list {
        let dep_data = DeploymentTrainingData::from_deployment(deployment);
        training_data.push(TrainingExample::from_deployment(dep_data));
    }
    
    let output_path = cli.output_dir.join("deployments_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Collected {} deployment examples", training_data.len());
    
    Ok(())
}

async fn collect_events(client: &Client, cli: &Cli, problems_only: bool, hours: u64) -> Result<()> {
    info!("🔍 Collecting event data (last {} hours)...", hours);
    
    let events: Api<Event> = if let Some(ns) = &cli.namespace {
        Api::namespaced(client.clone(), ns)
    } else {
        Api::all(client.clone())
    };

    let lp = ListParams::default();
    let event_list = events.list(&lp).await?;
    
    let cutoff_time = chrono::Utc::now() - chrono::Duration::hours(hours as i64);
    let mut training_data = Vec::new();
    
    for event in event_list {
        let event_data = EventTrainingData::from_event(event, cutoff_time);
        
        if let Some(data) = event_data {
            if problems_only && !data.is_problem() {
                continue;
            }
            
            training_data.push(TrainingExample::from_event(data));
        }
    }
    
    let output_path = cli.output_dir.join("events_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Collected {} event examples", training_data.len());
    
    Ok(())
}

async fn collect_nodes(client: &Client, cli: &Cli) -> Result<()> {
    info!("🔍 Collecting node data...");
    
    let nodes: Api<Node> = Api::all(client.clone());
    let lp = ListParams::default();
    let node_list = nodes.list(&lp).await?;
    
    let mut training_data = Vec::new();
    
    for node in node_list {
        let node_data = NodeTrainingData::from_node(node);
        training_data.push(TrainingExample::from_node(node_data));
    }
    
    let output_path = cli.output_dir.join("nodes_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Collected {} node examples", training_data.len());
    
    Ok(())
}

async fn collect_statefulsets(client: &Client, cli: &Cli) -> Result<()> {
    info!("🔍 Collecting StatefulSet data...");
    
    let statefulsets: Api<StatefulSet> = if let Some(ns) = &cli.namespace {
        Api::namespaced(client.clone(), ns)
    } else {
        Api::all(client.clone())
    };

    let lp = ListParams::default();
    let sts_list = statefulsets.list(&lp).await?;
    
    let mut training_data = Vec::new();
    
    for sts in sts_list {
        let sts_data = StatefulSetTrainingData::from_statefulset(sts);
        training_data.push(TrainingExample::from_statefulset(sts_data));
    }
    
    let output_path = cli.output_dir.join("statefulsets_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Collected {} StatefulSet examples", training_data.len());
    
    Ok(())
}

async fn collect_daemonsets(client: &Client, cli: &Cli) -> Result<()> {
    info!("🔍 Collecting DaemonSet data...");
    
    let daemonsets: Api<DaemonSet> = if let Some(ns) = &cli.namespace {
        Api::namespaced(client.clone(), ns)
    } else {
        Api::all(client.clone())
    };

    let lp = ListParams::default();
    let ds_list = daemonsets.list(&lp).await?;
    
    let mut training_data = Vec::new();
    
    for ds in ds_list {
        let ds_data = DaemonSetTrainingData::from_daemonset(ds);
        training_data.push(TrainingExample::from_daemonset(ds_data));
    }
    
    let output_path = cli.output_dir.join("daemonsets_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Collected {} DaemonSet examples", training_data.len());
    
    Ok(())
}

async fn collect_jobs(client: &Client, cli: &Cli) -> Result<()> {
    info!("🔍 Collecting Job data...");
    
    let jobs: Api<Job> = if let Some(ns) = &cli.namespace {
        Api::namespaced(client.clone(), ns)
    } else {
        Api::all(client.clone())
    };

    let lp = ListParams::default();
    let job_list = jobs.list(&lp).await?;
    
    let mut training_data = Vec::new();
    
    for job in job_list {
        let job_data = JobTrainingData::from_job(job);
        training_data.push(TrainingExample::from_job(job_data));
    }
    
    let output_path = cli.output_dir.join("jobs_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Collected {} Job examples", training_data.len());
    
    Ok(())
}

async fn collect_services(client: &Client, cli: &Cli) -> Result<()> {
    info!("🔍 Collecting Service data...");
    
    let services: Api<Service> = if let Some(ns) = &cli.namespace {
        Api::namespaced(client.clone(), ns)
    } else {
        Api::all(client.clone())
    };

    let lp = ListParams::default();
    let svc_list = services.list(&lp).await?;
    
    let mut training_data = Vec::new();
    
    for svc in svc_list {
        let svc_data = ServiceTrainingData::from_service(svc);
        training_data.push(TrainingExample::from_service(svc_data));
    }
    
    let output_path = cli.output_dir.join("services_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Collected {} Service examples", training_data.len());
    
    Ok(())
}

async fn collect_pvcs(client: &Client, cli: &Cli) -> Result<()> {
    info!("🔍 Collecting PVCs data...");
    
    let pvcs: Api<PersistentVolumeClaim> = if let Some(ns) = &cli.namespace {
        Api::namespaced(client.clone(), ns)
    } else {
        Api::all(client.clone())
    };

    let lp = ListParams::default();
    let pvc_list = pvcs.list(&lp).await?;
    
    let mut training_data = Vec::new();
    
    for pvc in pvc_list {
        let pvc_data = PvcTrainingData::from_pvc(pvc);
        training_data.push(TrainingExample::from_pvc(pvc_data));
    }
    
    let output_path = cli.output_dir.join("pvcs_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Collected {} PVC examples", training_data.len());
    
    Ok(())
}

async fn collect_pvs(client: &Client, cli: &Cli) -> Result<()> {
    info!("🔍 Collecting PVs data...");
    
    // PVs are cluster-scoped, so we always use Api::all()
    let pvs: Api<PersistentVolume> = Api::all(client.clone());

    let lp = ListParams::default();
    let pv_list = pvs.list(&lp).await?;
    
    let mut training_data = Vec::new();
    
    for pv in pv_list {
        let pv_data = PvTrainingData::from_pv(pv);
        training_data.push(TrainingExample::from_pv(pv_data));
    }
    
    let output_path = cli.output_dir.join("pvs_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Collected {} PV examples", training_data.len());
    
    Ok(())
}

async fn collect_storageclasses(client: &Client, cli: &Cli) -> Result<()> {
    info!("🔍 Collecting StorageClass data...");
    
    // StorageClasses are cluster-scoped, so we always use Api::all()
    let storageclasses: Api<StorageClass> = Api::all(client.clone());

    let lp = ListParams::default();
    let sc_list = storageclasses.list(&lp).await?;
    
    let mut training_data = Vec::new();
    
    for sc in sc_list {
        let sc_data = StorageClassTrainingData::from_storageclass(sc);
        training_data.push(TrainingExample::from_storageclass(sc_data));
    }
    
    let output_path = cli.output_dir.join("storageclasses_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Collected {} StorageClass examples", training_data.len());
    
    Ok(())
}

async fn generate_synthetic_data(cli: &Cli, count: usize) -> Result<()> {
    info!("🎲 Generating {} synthetic SRE scenarios...", count);
    
    let training_data = generate_synthetic_scenarios(count);
    
    let output_path = cli.output_dir.join("synthetic_training_data");
    export_training_data(&training_data, &output_path, &cli.format)?;
    
    info!("✅ Generated {} synthetic examples", training_data.len());
    
    Ok(())
}