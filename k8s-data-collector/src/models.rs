use chrono::{DateTime, Utc};
use k8s_openapi::api::{
    apps::v1::Deployment,
    core::v1::{Event, Node, Pod},
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
        let timestamp = event.last_timestamp
            .or(event.event_time)
            .and_then(|t| DateTime::parse_from_rfc3339(&t.0).ok())
            .map(|t| t.with_timezone(&Utc))?;
        
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
        let spec = node.spec.unwrap_or_default();
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
        
        Self {
            id,
            resource_type: "event".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: Some(event_data.namespace),
                cluster: None,
                severity: if event_data.is_problem() {
                    Severity::Warning
                } else {
                    Severity::Info
                },
                tags: vec!["kubernetes".to_string(), "event".to_string()],
            },
            timestamp: event_data.timestamp,
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
}