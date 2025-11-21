use chrono::{DateTime, Utc};
use k8s_openapi::api::{
    apps::v1::{Deployment, StatefulSet, DaemonSet},
    core::v1::{Event, Node, Pod, Service, PersistentVolumeClaim, PersistentVolume},
    batch::v1::Job,
    storage::v1::StorageClass,
};
use serde::{Deserialize, Serialize};

/// Training example for the SRE AI model
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingExample {
    /// Unique identifier
    pub id: String,
    
    /// Type of resource (pod, deployment, event, etc.)
    pub resource_type: String,
    
    /// The context/input for the model
    pub input: String,
    
    /// The expected output/diagnosis
    pub output: String,
    
    /// Additional metadata
    pub metadata: TrainingMetadata,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingMetadata {
    pub namespace: Option<String>,
    pub cluster: Option<String>,
    pub severity: Severity,
    pub tags: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Severity {
    Critical,
    Warning,
    Info,
    Normal,
}

/// Pod-specific training data
#[derive(Debug, Serialize, Deserialize)]
pub struct PodTrainingData {
    pub name: String,
    pub namespace: String,
    pub status: String,
    pub restart_count: i32,
    pub containers: Vec<ContainerInfo>,
    pub conditions: Vec<String>,
    pub resource_requests: ResourceInfo,
    pub resource_limits: ResourceInfo,
    pub events: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ContainerInfo {
    pub name: String,
    pub image: String,
    pub ready: bool,
    pub restart_count: i32,
    pub state: String,
    pub last_exit_code: Option<i32>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ResourceInfo {
    pub cpu: Option<String>,
    pub memory: Option<String>,
}

impl PodTrainingData {
    pub fn from_pod(pod: Pod) -> Self {
        let metadata = pod.metadata;
        let spec = pod.spec.unwrap_or_default();
        let status = pod.status.unwrap_or_default();
        
        let name = metadata.name.unwrap_or_default();
        let namespace = metadata.namespace.unwrap_or_else(|| "default".to_string());
        
        let pod_status = status.phase.unwrap_or_else(|| "Unknown".to_string());
        
        let containers: Vec<ContainerInfo> = status
            .container_statuses
            .unwrap_or_default()
            .into_iter()
            .map(|cs| {
                let state = if cs.state.is_some() {
                    let state = cs.state.unwrap();
                    if state.running.is_some() {
                        "Running".to_string()
                    } else if state.waiting.is_some() {
                        format!("Waiting: {}", state.waiting.unwrap().reason.unwrap_or_default())
                    } else if state.terminated.is_some() {
                        let term = state.terminated.unwrap();
                        format!("Terminated: {}", term.reason.unwrap_or_default())
                    } else {
                        "Unknown".to_string()
                    }
                } else {
                    "Unknown".to_string()
                };
                
                let last_exit_code = cs.last_state
                    .and_then(|s| s.terminated)
                    .map(|t| t.exit_code);
                
                ContainerInfo {
                    name: cs.name.clone(),
                    image: cs.image,
                    ready: cs.ready,
                    restart_count: cs.restart_count,
                    state,
                    last_exit_code,
                }
            })
            .collect();
        
        let restart_count = containers.iter().map(|c| c.restart_count).sum();
        
        let conditions: Vec<String> = status
            .conditions
            .unwrap_or_default()
            .into_iter()
            .filter(|c| c.status == "False")
            .map(|c| format!("{}: {}", c.type_, c.message.unwrap_or_default()))
            .collect();
        
        let (resource_requests, resource_limits) = spec
            .containers
            .get(0)
            .map(|c| {
                let requests = c.resources.as_ref()
                    .and_then(|r| r.requests.as_ref())
                    .map(|req| ResourceInfo {
                        cpu: req.get("cpu").map(|v| v.0.clone()),
                        memory: req.get("memory").map(|v| v.0.clone()),
                    })
                    .unwrap_or_default();
                
                let limits = c.resources.as_ref()
                    .and_then(|r| r.limits.as_ref())
                    .map(|lim| ResourceInfo {
                        cpu: lim.get("cpu").map(|v| v.0.clone()),
                        memory: lim.get("memory").map(|v| v.0.clone()),
                    })
                    .unwrap_or_default();
                
                (requests, limits)
            })
            .unwrap_or_default();
        
        Self {
            name,
            namespace,
            status: pod_status,
            restart_count,
            containers,
            conditions,
            resource_requests,
            resource_limits,
            events: Vec::new(),
        }
    }
    
    pub fn has_problems(&self) -> bool {
        self.status != "Running" 
            || self.restart_count > 3 
            || !self.conditions.is_empty()
            || self.containers.iter().any(|c| !c.ready)
    }
    
    pub fn generate_input(&self) -> String {
        format!(
            "Analyze this Kubernetes pod:\n\
            Name: {}\n\
            Namespace: {}\n\
            Status: {}\n\
            Restart Count: {}\n\
            Containers: {}\n\
            Issues: {}",
            self.name,
            self.namespace,
            self.status,
            self.restart_count,
            self.containers.len(),
            if self.conditions.is_empty() {
                "None".to_string()
            } else {
                self.conditions.join(", ")
            }
        )
    }
    
    pub fn generate_output(&self) -> String {
        let mut diagnosis = Vec::new();
        
        if self.status != "Running" {
            diagnosis.push(format!("Pod is in {} state - investigate pod events and logs", self.status));
        }
        
        if self.restart_count > 10 {
            diagnosis.push("High restart count indicates crashlooping - check application logs for errors".to_string());
        } else if self.restart_count > 3 {
            diagnosis.push("Elevated restart count - monitor for stability issues".to_string());
        }
        
        for container in &self.containers {
            if !container.ready {
                diagnosis.push(format!(
                    "Container '{}' not ready - current state: {}",
                    container.name, container.state
                ));
            }
            
            if let Some(exit_code) = container.last_exit_code {
                if exit_code != 0 {
                    diagnosis.push(format!(
                        "Container '{}' exited with code {} - check logs for error details",
                        container.name, exit_code
                    ));
                }
            }
        }
        
        for condition in &self.conditions {
            diagnosis.push(format!("Pod condition: {}", condition));
        }
        
        if diagnosis.is_empty() {
            "Pod appears healthy - no immediate issues detected".to_string()
        } else {
            format!(
                "Diagnosis:\n{}",
                diagnosis.iter().enumerate()
                    .map(|(i, d)| format!("{}. {}", i + 1, d))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        }
    }
}

/// Deployment-specific training data
#[derive(Debug, Serialize, Deserialize)]
pub struct DeploymentTrainingData {
    pub name: String,
    pub namespace: String,
    pub replicas_desired: i32,
    pub replicas_ready: i32,
    pub replicas_available: i32,
    pub strategy: String,
    pub conditions: Vec<String>,
}

impl DeploymentTrainingData {
    pub fn from_deployment(deployment: Deployment) -> Self {
        let metadata = deployment.metadata;
        let spec = deployment.spec.unwrap_or_default();
        let status = deployment.status.unwrap_or_default();
        
        let name = metadata.name.unwrap_or_default();
        let namespace = metadata.namespace.unwrap_or_else(|| "default".to_string());
        let replicas_desired = spec.replicas.unwrap_or(1);
        
        let replicas_ready = status.ready_replicas.unwrap_or(0);
        let replicas_available = status.available_replicas.unwrap_or(0);
        
        let strategy = spec.strategy
            .and_then(|s| s.type_)
            .unwrap_or_else(|| "RollingUpdate".to_string());
        
        let conditions: Vec<String> = status
            .conditions
            .unwrap_or_default()
            .into_iter()
            .filter(|c| c.status == "False")
            .map(|c| format!("{}: {}", c.type_, c.message.unwrap_or_default()))
            .collect();
        
        Self {
            name,
            namespace,
            replicas_desired,
            replicas_ready,
            replicas_available,
            strategy,
            conditions,
        }
    }
}

/// StatefulSet-specific training data
#[derive(Debug, Serialize, Deserialize)]
pub struct StatefulSetTrainingData {
    pub name: String,
    pub namespace: String,
    pub replicas_desired: i32,
    pub replicas_ready: i32,
    pub replicas_current: i32,
    pub service_name: String,
    pub conditions: Vec<String>,
}

impl StatefulSetTrainingData {
    pub fn from_statefulset(sts: StatefulSet) -> Self {
        let metadata = sts.metadata;
        let spec = sts.spec.unwrap_or_default();
        let status = sts.status.unwrap_or_default();
        
        let name = metadata.name.unwrap_or_default();
        let namespace = metadata.namespace.unwrap_or_else(|| "default".to_string());
        let replicas_desired = spec.replicas.unwrap_or(1);
        let service_name = spec.service_name;
        
        let replicas_ready = status.ready_replicas.unwrap_or(0);
        let replicas_current = status.current_replicas.unwrap_or(0);
        
        let conditions: Vec<String> = vec![];
        
        Self {
            name,
            namespace,
            replicas_desired,
            replicas_ready,
            replicas_current,
            service_name,
            conditions,
        }
    }
}

/// DaemonSet-specific training data
#[derive(Debug, Serialize, Deserialize)]
pub struct DaemonSetTrainingData {
    pub name: String,
    pub namespace: String,
    pub desired_scheduled: i32,
    pub current_scheduled: i32,
    pub number_ready: i32,
    pub number_available: i32,
    pub number_misscheduled: i32,
    pub conditions: Vec<String>,
}

impl DaemonSetTrainingData {
    pub fn from_daemonset(ds: DaemonSet) -> Self {
        let metadata = ds.metadata;
        let status = ds.status.unwrap_or_default();
        
        let name = metadata.name.unwrap_or_default();
        let namespace = metadata.namespace.unwrap_or_else(|| "default".to_string());
        
        let desired_scheduled = status.desired_number_scheduled;
        let current_scheduled = status.current_number_scheduled;
        let number_ready = status.number_ready;
        let number_available = status.number_available.unwrap_or(0);
        let number_misscheduled = status.number_misscheduled;
        
        let conditions: Vec<String> = status
            .conditions
            .unwrap_or_default()
            .into_iter()
            .filter(|c| c.status == "False")
            .map(|c| format!("{}: {}", c.type_, c.message.unwrap_or_default()))
            .collect();
        
        Self {
            name,
            namespace,
            desired_scheduled,
            current_scheduled,
            number_ready,
            number_available,
            number_misscheduled,
            conditions,
        }
    }
}

/// Job-specific training data
#[derive(Debug, Serialize, Deserialize)]
pub struct JobTrainingData {
    pub name: String,
    pub namespace: String,
    pub completions: i32,
    pub parallelism: i32,
    pub succeeded: i32,
    pub failed: i32,
    pub active: i32,
    pub conditions: Vec<String>,
}

impl JobTrainingData {
    pub fn from_job(job: Job) -> Self {
        let metadata = job.metadata;
        let spec = job.spec.unwrap_or_default();
        let status = job.status.unwrap_or_default();
        
        let name = metadata.name.unwrap_or_default();
        let namespace = metadata.namespace.unwrap_or_else(|| "default".to_string());
        
        let completions = spec.completions.unwrap_or(1);
        let parallelism = spec.parallelism.unwrap_or(1);
        
        let succeeded = status.succeeded.unwrap_or(0);
        let failed = status.failed.unwrap_or(0);
        let active = status.active.unwrap_or(0);
        
        let conditions: Vec<String> = status
            .conditions
            .unwrap_or_default()
            .into_iter()
            .filter(|c| c.status == "False")
            .map(|c| format!("{}: {}", c.type_, c.message.unwrap_or_default()))
            .collect();
        
        Self {
            name,
            namespace,
            completions,
            parallelism,
            succeeded,
            failed,
            active,
            conditions,
        }
    }
}

/// Service-specific training data
#[derive(Debug, Serialize, Deserialize)]
pub struct ServiceTrainingData {
    pub name: String,
    pub namespace: String,
    pub service_type: String,
    pub cluster_ip: String,
    pub external_ips: Vec<String>,
    pub ports: Vec<String>,
    pub selector: Vec<String>,
}

impl ServiceTrainingData {
    pub fn from_service(svc: Service) -> Self {
        let metadata = svc.metadata;
        let spec = svc.spec.unwrap_or_default();
        
        let name = metadata.name.unwrap_or_default();
        let namespace = metadata.namespace.unwrap_or_else(|| "default".to_string());
        
        let service_type = spec.type_.unwrap_or_else(|| "ClusterIP".to_string());
        let cluster_ip = spec.cluster_ip.unwrap_or_default();
        let external_ips = spec.external_ips.unwrap_or_default();
        
        let ports: Vec<String> = spec
            .ports
            .unwrap_or_default()
            .into_iter()
            .map(|p| {
                format!(
                    "{}:{}/{}",
                    p.name.unwrap_or_default(),
                    p.port,
                    p.protocol.unwrap_or_else(|| "TCP".to_string())
                )
            })
            .collect();
        
        let selector: Vec<String> = spec
            .selector
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        
        Self {
            name,
            namespace,
            service_type,
            cluster_ip,
            external_ips,
            ports,
            selector,
        }
    }
}

/// PVC-specific training data
#[derive(Debug, Serialize, Deserialize)]
pub struct PvcTrainingData {
    pub name: String,
    pub namespace: String,
    pub status: String,
    pub storage_class: String,
    pub capacity: String,
    pub access_modes: Vec<String>,
    pub volume_name: String,
}

impl PvcTrainingData {
    pub fn from_pvc(pvc: PersistentVolumeClaim) -> Self {
        let metadata = pvc.metadata;
        let spec = pvc.spec.unwrap_or_default();
        let status = pvc.status.unwrap_or_default();
        
        let name = metadata.name.unwrap_or_default();
        let namespace = metadata.namespace.unwrap_or_else(|| "default".to_string());
        
        let pvc_status = status.phase.unwrap_or_else(|| "Unknown".to_string());
        let storage_class = spec.storage_class_name.unwrap_or_else(|| "default".to_string());
        
        let capacity = status
            .capacity
            .and_then(|c| c.get("storage").map(|v| v.0.clone()))
            .unwrap_or_else(|| "unknown".to_string());
        
        let access_modes = spec.access_modes.unwrap_or_default();
        let volume_name = spec.volume_name.unwrap_or_default();
        
        Self {
            name,
            namespace,
            status: pvc_status,
            storage_class,
            capacity,
            access_modes,
            volume_name,
        }
    }
}

/// PV-specific training data
#[derive(Debug, Serialize, Deserialize)]
pub struct PvTrainingData {
    pub name: String,
    pub status: String,
    pub storage_class: String,
    pub capacity: String,
    pub access_modes: Vec<String>,
    pub reclaim_policy: String,
    pub claim_ref: Option<String>,
    pub volume_mode: String,
}

impl PvTrainingData {
    pub fn from_pv(pv: PersistentVolume) -> Self {
        let metadata = pv.metadata;
        let spec = pv.spec.unwrap_or_default();
        let status = pv.status.unwrap_or_default();
        
        let name = metadata.name.unwrap_or_default();
        let pv_status = status.phase.unwrap_or_else(|| "Unknown".to_string());
        let storage_class = spec.storage_class_name.unwrap_or_else(|| "default".to_string());
        
        let capacity = spec
            .capacity
            .and_then(|c| c.get("storage").map(|v| v.0.clone()))
            .unwrap_or_else(|| "unknown".to_string());
        
        let access_modes = spec.access_modes.unwrap_or_default();
        let reclaim_policy = spec.persistent_volume_reclaim_policy.unwrap_or_else(|| "Retain".to_string());
        let volume_mode = spec.volume_mode.unwrap_or_else(|| "Filesystem".to_string());
        
        let claim_ref = spec.claim_ref.and_then(|cr| {
            cr.namespace.and_then(|ns| cr.name.map(|name| format!("{}/{}", ns, name)))
        });
        
        Self {
            name,
            status: pv_status,
            storage_class,
            capacity,
            access_modes,
            reclaim_policy,
            claim_ref,
            volume_mode,
        }
    }
}

/// StorageClass-specific training data
#[derive(Debug, Serialize, Deserialize)]
pub struct StorageClassTrainingData {
    pub name: String,
    pub provisioner: String,
    pub reclaim_policy: String,
    pub volume_binding_mode: String,
    pub allow_volume_expansion: bool,
    pub parameters: Vec<String>,
}

impl StorageClassTrainingData {
    pub fn from_storageclass(sc: StorageClass) -> Self {
        let metadata = sc.metadata;
        
        let name = metadata.name.unwrap_or_default();
        let provisioner = sc.provisioner;
        let reclaim_policy = sc.reclaim_policy.unwrap_or_else(|| "Delete".to_string());
        let volume_binding_mode = sc.volume_binding_mode.unwrap_or_else(|| "Immediate".to_string());
        let allow_volume_expansion = sc.allow_volume_expansion.unwrap_or(false);
        
        let parameters: Vec<String> = sc
            .parameters
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect();
        
        Self {
            name,
            provisioner,
            reclaim_policy,
            volume_binding_mode,
            allow_volume_expansion,
            parameters,
        }
    }
}

/// Event-specific training data
#[derive(Debug, Serialize, Deserialize)]
pub struct EventTrainingData {
    pub reason: String,
    pub message: String,
    pub type_: String,
    pub involved_object: String,
    pub namespace: String,
    pub timestamp: DateTime<Utc>,
}

impl EventTrainingData {
    pub fn from_event(event: Event, cutoff: DateTime<Utc>) -> Option<Self> {
        let metadata = event.metadata;
        
        let timestamp = if let Some(last_ts) = event.last_timestamp {
            last_ts.0
        } else if let Some(event_time) = event.event_time {
            event_time.0
        } else {
            return None;
        };
        
        if timestamp < cutoff {
            return None;
        }
        
        Some(Self {
            reason: event.reason.unwrap_or_default(),
            message: event.message.unwrap_or_default(),
            type_: event.type_.unwrap_or_default(),
            involved_object: event.involved_object.name.unwrap_or_default(),
            namespace: metadata.namespace.unwrap_or_default(),
            timestamp,
        })
    }
    
    pub fn is_problem(&self) -> bool {
        self.type_ == "Warning" || self.type_ == "Error"
    }
}

/// Node-specific training data
#[derive(Debug, Serialize, Deserialize)]
pub struct NodeTrainingData {
    pub name: String,
    pub capacity: ResourceInfo,
    pub allocatable: ResourceInfo,
    pub conditions: Vec<String>,
    pub kubelet_version: String,
}

impl NodeTrainingData {
    pub fn from_node(node: Node) -> Self {
        let metadata = node.metadata;
        let status = node.status.unwrap_or_default();
        
        let name = metadata.name.unwrap_or_default();
        
        let capacity = status.capacity
            .map(|cap| ResourceInfo {
                cpu: cap.get("cpu").map(|v| v.0.clone()),
                memory: cap.get("memory").map(|v| v.0.clone()),
            })
            .unwrap_or_default();
        
        let allocatable = status.allocatable
            .map(|alloc| ResourceInfo {
                cpu: alloc.get("cpu").map(|v| v.0.clone()),
                memory: alloc.get("memory").map(|v| v.0.clone()),
            })
            .unwrap_or_default();
        
        let conditions: Vec<String> = status
            .conditions
            .unwrap_or_default()
            .into_iter()
            .filter(|c| {
                (c.type_ == "Ready" && c.status == "False")
                    || (c.type_ != "Ready" && c.status == "True")
            })
            .map(|c| format!("{}: {}", c.type_, c.message.unwrap_or_default()))
            .collect();
        
        let kubelet_version = status.node_info
            .map(|info| info.kubelet_version)
            .unwrap_or_default();
        
        Self {
            name,
            capacity,
            allocatable,
            conditions,
            kubelet_version,
        }
    }
}

impl TrainingExample {
    pub fn from_pod(pod_data: PodTrainingData) -> Self {
        let id = format!("pod-{}-{}", pod_data.namespace, pod_data.name);
        let severity = if pod_data.status == "Running" && pod_data.restart_count < 3 {
            Severity::Normal
        } else if pod_data.restart_count > 10 {
            Severity::Critical
        } else {
            Severity::Warning
        };
        
        Self {
            id,
            resource_type: "statefulset".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: Some(sts_data.namespace),
                cluster: None,
                severity: if sts_data.replicas_ready < sts_data.replicas_desired {
                    Severity::Warning
                } else {
                    Severity::Normal
                },
                tags: vec!["kubernetes".to_string(), "statefulset".to_string()],
            },
            timestamp: Utc::now(),
        }
    }
    
    pub fn from_daemonset(ds_data: DaemonSetTrainingData) -> Self {
        let id = format!("daemonset-{}-{}", ds_data.namespace, ds_data.name);
        
        let input = format!(
            "Analyze this Kubernetes DaemonSet:\n\
            Name: {}\n\
            Namespace: {}\n\
            Desired Scheduled: {}\n\
            Current Scheduled: {}\n\
            Ready: {}\n\
            Available: {}\n\
            Misscheduled: {}",
            ds_data.name,
            ds_data.namespace,
            ds_data.desired_scheduled,
            ds_data.current_scheduled,
            ds_data.number_ready,
            ds_data.number_available,
            ds_data.number_misscheduled
        );
        
        let mut issues = Vec::new();
        
        if ds_data.number_ready < ds_data.desired_scheduled {
            issues.push(format!(
                "Only {}/{} pods are ready",
                ds_data.number_ready, ds_data.desired_scheduled
            ));
        }
        
        if ds_data.number_misscheduled > 0 {
            issues.push(format!(
                "{} pods are misscheduled - check node selectors and tolerations",
                ds_data.number_misscheduled
            ));
        }
        
        let output = if issues.is_empty() {
            "DaemonSet is healthy - all desired pods are running".to_string()
        } else {
            format!("DaemonSet issues:\n{}", issues.join("\n"))
        };
        
        Self {
            id,
            resource_type: "daemonset".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: Some(ds_data.namespace),
                cluster: None,
                severity: if !issues.is_empty() {
                    Severity::Warning
                } else {
                    Severity::Normal
                },
                tags: vec!["kubernetes".to_string(), "daemonset".to_string()],
            },
            timestamp: Utc::now(),
        }
    }
    
    pub fn from_job(job_data: JobTrainingData) -> Self {
        let id = format!("job-{}-{}", job_data.namespace, job_data.name);
        
        let input = format!(
            "Analyze this Kubernetes Job:\n\
            Name: {}\n\
            Namespace: {}\n\
            Completions: {}\n\
            Parallelism: {}\n\
            Succeeded: {}\n\
            Failed: {}\n\
            Active: {}",
            job_data.name,
            job_data.namespace,
            job_data.completions,
            job_data.parallelism,
            job_data.succeeded,
            job_data.failed,
            job_data.active
        );
        
        let output = if job_data.succeeded >= job_data.completions {
            "Job completed successfully".to_string()
        } else if job_data.failed > 0 {
            format!(
                "Job has {} failed pods - check logs for errors. Succeeded: {}/{}",
                job_data.failed, job_data.succeeded, job_data.completions
            )
        } else if job_data.active > 0 {
            format!(
                "Job is running - {}/{} completions, {} active pods",
                job_data.succeeded, job_data.completions, job_data.active
            )
        } else {
            "Job status unknown - investigate pod states".to_string()
        };
        
        Self {
            id,
            resource_type: "job".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: Some(job_data.namespace),
                cluster: None,
                severity: if job_data.failed > 0 {
                    Severity::Warning
                } else if job_data.succeeded >= job_data.completions {
                    Severity::Normal
                } else {
                    Severity::Info
                },
                tags: vec!["kubernetes".to_string(), "job".to_string()],
            },
            timestamp: Utc::now(),
        }
    }
    
    pub fn from_service(svc_data: ServiceTrainingData) -> Self {
        let id = format!("service-{}-{}", svc_data.namespace, svc_data.name);
        
        let input = format!(
            "Analyze this Kubernetes Service:\n\
            Name: {}\n\
            Namespace: {}\n\
            Type: {}\n\
            ClusterIP: {}\n\
            Ports: {}\n\
            Selector: {}",
            svc_data.name,
            svc_data.namespace,
            svc_data.service_type,
            svc_data.cluster_ip,
            svc_data.ports.join(", "),
            svc_data.selector.join(", ")
        );
        
        let output = if svc_data.selector.is_empty() {
            "Service has no selector - it won't route traffic to any pods".to_string()
        } else {
            format!(
                "Service is configured to route traffic to pods matching: {}",
                svc_data.selector.join(", ")
            )
        };
        
        Self {
            id,
            resource_type: "service".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: Some(svc_data.namespace),
                cluster: None,
                severity: if svc_data.selector.is_empty() {
                    Severity::Warning
                } else {
                    Severity::Normal
                },
                tags: vec!["kubernetes".to_string(), "service".to_string()],
            },
            timestamp: Utc::now(),
        }
    }
    
    pub fn from_pvc(pvc_data: PvcTrainingData) -> Self {
        let id = format!("pvc-{}-{}", pvc_data.namespace, pvc_data.name);
        
        let input = format!(
            "Analyze this Kubernetes PersistentVolumeClaim:\n\
            Name: {}\n\
            Namespace: {}\n\
            Status: {}\n\
            Storage Class: {}\n\
            Capacity: {}\n\
            Access Modes: {}",
            pvc_data.name,
            pvc_data.namespace,
            pvc_data.status,
            pvc_data.storage_class,
            pvc_data.capacity,
            pvc_data.access_modes.join(", ")
        );
        
        let output = match pvc_data.status.as_str() {
            "Bound" => "PVC is bound to a volume and ready to use".to_string(),
            "Pending" => "PVC is pending - check storage class and available PVs".to_string(),
            "Lost" => "PVC has lost its underlying volume - data may be lost".to_string(),
            _ => format!("PVC status: {} - investigate storage configuration", pvc_data.status),
        };
        
        Self {
            id,
            resource_type: "pvc".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: Some(pvc_data.namespace),
                cluster: None,
                severity: match pvc_data.status.as_str() {
                    "Bound" => Severity::Normal,
                    "Lost" => Severity::Critical,
                    _ => Severity::Warning,
                },
                tags: vec!["kubernetes".to_string(), "pvc".to_string(), "storage".to_string()],
            },
            timestamp: Utc::now(),
        }
    }
    
    pub fn from_pv(pv_data: PvTrainingData) -> Self {
        let id = format!("pv-{}", pv_data.name);
        
        let input = format!(
            "Analyze this Kubernetes PersistentVolume:\n\
            Name: {}\n\
            Status: {}\n\
            Storage Class: {}\n\
            Capacity: {}\n\
            Access Modes: {}\n\
            Reclaim Policy: {}\n\
            Volume Mode: {}\n\
            Claim: {}",
            pv_data.name,
            pv_data.status,
            pv_data.storage_class,
            pv_data.capacity,
            pv_data.access_modes.join(", "),
            pv_data.reclaim_policy,
            pv_data.volume_mode,
            pv_data.claim_ref.as_deref().unwrap_or("None")
        );
        
        let output = match pv_data.status.as_str() {
            "Available" => "PV is available and ready to be claimed".to_string(),
            "Bound" => format!(
                "PV is bound to claim: {}",
                pv_data.claim_ref.as_deref().unwrap_or("unknown")
            ),
            "Released" => "PV was released from claim but not yet reclaimed - check reclaim policy".to_string(),
            "Failed" => "PV has failed - investigate underlying storage issues".to_string(),
            _ => format!("PV status: {} - investigate storage configuration", pv_data.status),
        };
        
        Self {
            id,
            resource_type: "pv".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: None,
                cluster: None,
                severity: match pv_data.status.as_str() {
                    "Available" | "Bound" => Severity::Normal,
                    "Failed" => Severity::Critical,
                    _ => Severity::Warning,
                },
                tags: vec!["kubernetes".to_string(), "pv".to_string(), "storage".to_string()],
            },
            timestamp: Utc::now(),
        }
    }
    
    pub fn from_storageclass(sc_data: StorageClassTrainingData) -> Self {
        let id = format!("storageclass-{}", sc_data.name);
        
        let input = format!(
            "Analyze this Kubernetes StorageClass:\n\
            Name: {}\n\
            Provisioner: {}\n\
            Reclaim Policy: {}\n\
            Volume Binding Mode: {}\n\
            Allow Volume Expansion: {}\n\
            Parameters: {}",
            sc_data.name,
            sc_data.provisioner,
            sc_data.reclaim_policy,
            sc_data.volume_binding_mode,
            sc_data.allow_volume_expansion,
            if sc_data.parameters.is_empty() {
                "None".to_string()
            } else {
                sc_data.parameters.join(", ")
            }
        );
        
        let mut analysis = Vec::new();
        
        analysis.push(format!(
            "StorageClass uses {} provisioner for dynamic volume provisioning",
            sc_data.provisioner
        ));
        
        if sc_data.reclaim_policy == "Delete" {
            analysis.push("Volumes will be automatically deleted when PVC is deleted".to_string());
        } else {
            analysis.push("Volumes will be retained after PVC deletion - manual cleanup required".to_string());
        }
        
        if sc_data.volume_binding_mode == "WaitForFirstConsumer" {
            analysis.push("Volume binding delayed until pod is scheduled - better for topology-aware provisioning".to_string());
        } else {
            analysis.push("Volume binding happens immediately when PVC is created".to_string());
        }
        
        if sc_data.allow_volume_expansion {
            analysis.push("Volume expansion is enabled - PVCs can be resized".to_string());
        } else {
            analysis.push("Volume expansion is disabled - PVCs cannot be resized".to_string());
        }
        
        let output = format!("StorageClass analysis:\n{}", analysis.join("\n"));
        
        Self {
            id,
            resource_type: "storageclass".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: None,
                cluster: None,
                severity: Severity::Info,
                tags: vec!["kubernetes".to_string(), "storageclass".to_string(), "storage".to_string()],
            },
            timestamp: Utc::now(),
        }
    }d,
            resource_type: "pod".to_string(),
            input: pod_data.generate_input(),
            output: pod_data.generate_output(),
            metadata: TrainingMetadata {
                namespace: Some(pod_data.namespace.clone()),
                cluster: None,
                severity,
                tags: vec!["kubernetes".to_string(), "pod".to_string()],
            },
            timestamp: Utc::now(),
        }
    }
    
    pub fn from_deployment(dep_data: DeploymentTrainingData) -> Self {
        let id = format!("deployment-{}-{}", dep_data.namespace, dep_data.name);
        
        let input = format!(
            "Analyze this Kubernetes deployment:\n\
            Name: {}\n\
            Namespace: {}\n\
            Desired Replicas: {}\n\
            Ready Replicas: {}\n\
            Available Replicas: {}",
            dep_data.name,
            dep_data.namespace,
            dep_data.replicas_desired,
            dep_data.replicas_ready,
            dep_data.replicas_available
        );
        
        let output = if dep_data.replicas_ready == dep_data.replicas_desired {
            "Deployment is healthy - all replicas are ready".to_string()
        } else {
            format!(
                "Deployment has {}/{} replicas ready - investigate pod issues",
                dep_data.replicas_ready, dep_data.replicas_desired
            )
        };
        
        Self {
            id,
            resource_type: "deployment".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: Some(dep_data.namespace),
                cluster: None,
                severity: if dep_data.replicas_ready < dep_data.replicas_desired {
                    Severity::Warning
                } else {
                    Severity::Normal
                },
                tags: vec!["kubernetes".to_string(), "deployment".to_string()],
            },
            timestamp: Utc::now(),
        }
    }
    
    pub fn from_event(event_data: EventTrainingData) -> Self {
        let id = format!("event-{}-{}", event_data.namespace, event_data.timestamp.timestamp());
        
        let input = format!(
            "Kubernetes event:\n\
            Type: {}\n\
            Reason: {}\n\
            Object: {}\n\
            Message: {}",
            event_data.type_,
            event_data.reason,
            event_data.involved_object,
            event_data.message
        );
        
        let output = format!(
            "Event analysis: {} event for {} - {}",
            event_data.type_,
            event_data.involved_object,
            event_data.message
        );
        
        let is_problem = event_data.is_problem();
        let timestamp = event_data.timestamp;
        let namespace = event_data.namespace;
        
        Self {
            id,
            resource_type: "event".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: Some(namespace),
                cluster: None,
                severity: if is_problem {
                    Severity::Warning
                } else {
                    Severity::Info
                },
                tags: vec!["kubernetes".to_string(), "event".to_string()],
            },
            timestamp,
        }
    }
    
    pub fn from_node(node_data: NodeTrainingData) -> Self {
        let id = format!("node-{}", node_data.name);
        
        let input = format!(
            "Analyze this Kubernetes node:\n\
            Name: {}\n\
            CPU Capacity: {}\n\
            Memory Capacity: {}\n\
            Kubelet Version: {}",
            node_data.name,
            node_data.capacity.cpu.as_deref().unwrap_or("unknown"),
            node_data.capacity.memory.as_deref().unwrap_or("unknown"),
            node_data.kubelet_version
        );
        
        let output = if node_data.conditions.is_empty() {
            "Node is healthy - all conditions normal".to_string()
        } else {
            format!("Node issues detected:\n{}", node_data.conditions.join("\n"))
        };
        
        Self {
            id,
            resource_type: "node".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: None,
                cluster: None,
                severity: if node_data.conditions.is_empty() {
                    Severity::Normal
                } else {
                    Severity::Warning
                },
                tags: vec!["kubernetes".to_string(), "node".to_string()],
            },
            timestamp: Utc::now(),
        }
    }
    
    pub fn from_statefulset(sts_data: StatefulSetTrainingData) -> Self {
        let id = format!("statefulset-{}-{}", sts_data.namespace, sts_data.name);
        
        let input = format!(
            "Analyze this Kubernetes StatefulSet:\n\
            Name: {}\n\
            Namespace: {}\n\
            Desired Replicas: {}\n\
            Ready Replicas: {}\n\
            Current Replicas: {}\n\
            Service: {}",
            sts_data.name,
            sts_data.namespace,
            sts_data.replicas_desired,
            sts_data.replicas_ready,
            sts_data.replicas_current,
            sts_data.service_name
        );
        
        let output = if sts_data.replicas_ready == sts_data.replicas_desired {
            "StatefulSet is healthy - all replicas are ready".to_string()
        } else {
            format!(
                "StatefulSet has {}/{} replicas ready - investigate pod issues and check PVCs",
                sts_data.replicas_ready, sts_data.replicas_desired
            )
        };
}