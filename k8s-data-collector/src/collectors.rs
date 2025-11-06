use crate::models::*;
use rand::Rng;

/// Generate synthetic SRE scenarios for training
pub fn generate_synthetic_scenarios(count: usize) -> Vec<TrainingExample> {
    let mut scenarios = Vec::new();
    let mut rng = rand::thread_rng();
    
    let scenario_templates = vec![
        // OOMKilled scenarios
        SyntheticTemplate {
            input: "Pod experiencing OOMKilled errors. Container memory limit: {memory}. Last exit code: 137.",
            output: "Container exceeded memory limit of {memory}. Solution: 1) Increase memory limits, 2) Profile application memory usage, 3) Check for memory leaks.",
            severity: Severity::Critical,
            tags: vec!["oom", "memory"],
        },
        
        // CrashLoopBackOff
        SyntheticTemplate {
            input: "Pod in CrashLoopBackOff state. Restart count: {restarts}. Last error: {error}",
            output: "Application failing to start. Troubleshooting: 1) Check logs with kubectl logs, 2) Verify configuration and environment variables, 3) Check dependencies.",
            severity: Severity::Critical,
            tags: vec!["crashloop", "startup"],
        },
        
        // ImagePullBackOff
        SyntheticTemplate {
            input: "Pod stuck in ImagePullBackOff. Image: {image}. Error: Failed to pull image.",
            output: "Cannot pull container image. Solutions: 1) Verify image exists in registry, 2) Check image pull secrets, 3) Verify network connectivity to registry.",
            severity: Severity::Warning,
            tags: vec!["image", "registry"],
        },
        
        // High CPU usage
        SyntheticTemplate {
            input: "Node CPU usage at {cpu}%. Multiple pods throttled. Cluster autoscaling: {autoscale}",
            output: "High CPU utilization detected. Actions: 1) Review pod resource requests/limits, 2) Consider horizontal pod autoscaling, 3) Add nodes if autoscaling disabled.",
            severity: Severity::Warning,
            tags: vec!["cpu", "resources"],
        },
        
        // Disk pressure
        SyntheticTemplate {
            input: "Node experiencing disk pressure. Available disk: {disk}GB. Condition: DiskPressure=True",
            output: "Node running low on disk space. Immediate actions: 1) Clean up old container images, 2) Review persistent volume claims, 3) Implement log rotation.",
            severity: Severity::Critical,
            tags: vec!["disk", "storage"],
        },
        
        // Liveness probe failures
        SyntheticTemplate {
            input: "Pod failing liveness probe. Probe: {probe_type}. Timeout: {timeout}s. Failures: {failures}",
            output: "Health check failing. Investigation: 1) Verify endpoint responds within timeout, 2) Check application health, 3) Review probe configuration.",
            severity: Severity::Warning,
            tags: vec!["probe", "health"],
        },
        
        // Pending pods
        SyntheticTemplate {
            input: "Pod stuck in Pending state. Reason: {reason}. Resource requests: CPU={cpu}, Memory={memory}",
            output: "Pod cannot be scheduled. Common causes: 1) Insufficient cluster resources, 2) Node selector/affinity conflicts, 3) PVC binding issues.",
            severity: Severity::Warning,
            tags: vec!["scheduling", "pending"],
        },
        
        // Service connectivity
        SyntheticTemplate {
            input: "Service {service} returning 503 errors. Backend pods: {healthy}/{total} healthy. Load balancer status: {lb_status}",
            output: "Service degraded. Troubleshooting: 1) Check pod readiness probes, 2) Verify service selector matches pods, 3) Review endpoint configuration.",
            severity: Severity::Critical,
            tags: vec!["service", "networking"],
        },
    ];
    
    for i in 0..count {
        let template = &scenario_templates[rng.gen_range(0..scenario_templates.len())];
        
        let (input, output) = fill_template(template, &mut rng);
        
        scenarios.push(TrainingExample {
            id: format!("synthetic-{}", i),
            resource_type: "synthetic".to_string(),
            input,
            output,
            metadata: TrainingMetadata {
                namespace: Some(format!("prod-{}", rng.gen_range(1..5))),
                cluster: Some("training-cluster".to_string()),
                severity: template.severity.clone(),
                tags: template.tags.iter().map(|s| s.to_string()).collect(),
            },
            timestamp: chrono::Utc::now(),
        });
    }
    
    scenarios
}

struct SyntheticTemplate {
    input: &'static str,
    output: &'static str,
    severity: Severity,
    tags: Vec<&'static str>,
}

fn fill_template(template: &SyntheticTemplate, rng: &mut impl Rng) -> (String, String) {
    let replacements = vec![
        ("{memory}", vec!["256Mi", "512Mi", "1Gi", "2Gi", "4Gi"]),
        ("{restarts}", vec!["5", "10", "15", "20", "50"]),
        ("{error}", vec![
            "connection refused",
            "config file not found",
            "database connection failed",
            "port already in use",
        ]),
        ("{image}", vec![
            "registry.example.com/app:v1.2.3",
            "docker.io/myapp:latest",
            "gcr.io/project/service:sha256",
        ]),
        ("{cpu}", vec!["85", "90", "95", "98"]),
        ("{autoscale}", vec!["enabled", "disabled"]),
        ("{disk}", vec!["5", "3", "1", "0.5"]),
        ("{probe_type}", vec!["HTTP GET", "TCP", "Exec"]),
        ("{timeout}", vec!["3", "5", "10"]),
        ("{failures}", vec!["3", "5", "10"]),
        ("{reason}", vec![
            "Insufficient cpu",
            "Insufficient memory",
            "No nodes available",
            "PVC pending",
        ]),
        ("{service}", vec!["api-service", "web-frontend", "database-proxy"]),
        ("{healthy}", vec!["0", "1", "2"]),
        ("{total}", vec!["3", "5", "10"]),
        ("{lb_status}", vec!["healthy", "degraded", "unhealthy"]),
    ];
    
    let mut input = template.input.to_string();
    let mut output = template.output.to_string();
    
    for (placeholder, options) in replacements {
        let value = options[rng.gen_range(0..options.len())];
        input = input.replace(placeholder, value);
        output = output.replace(placeholder, value);
    }
    
    (input, output)
}