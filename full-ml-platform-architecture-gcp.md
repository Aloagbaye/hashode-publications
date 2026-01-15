---
title: Building a Full ML Platform on GCP: The Complete Reference Architecture
subtitle: "The definitive guide to designing and automating a production-grade ML platform on Google Cloud"
slug: building-full-ml-platform-on-gcp
cover_image: 
tags: machine-learning, Terraform, google-cloud, vertex-AI, Pub/Sub
domain: israelcodes.hashnode.dev
---

# Building a Full ML Platform on GCP: The Complete Reference Architecture
---

## Introduction

If you're building machine learning at scale in Google Cloud Platform (GCP), you need more than just a trained model. You need a **complete ML platform** that can handle the entire lifecycle—from data ingestion to model deployment, monitoring, and automated retraining.

This post walks through a production-ready ML platform architecture that answers the question: *"If I were building ML at scale in GCP, how do all the pieces fit together?"*

---

## 🧠 The High-Level Vision

A **Full ML Platform** is a system that can:

- ✅ **Ingest data automatically** from various sources
- ✅ **Trigger ML workflows via events** (no manual intervention)
- ✅ **Train models reproducibly** with full lineage tracking
- ✅ **Deploy models safely** with canary deployments and rollbacks
- ✅ **Monitor & retrain continuously** based on drift and performance
- ✅ **Be fully automated** with Infrastructure as Code (IaC) and CI/CD

All of this without clicking buttons in the console. Everything is code, version-controlled, and reproducible.

---

## 🧩 The Architecture: End-to-End Flow

Here's how the complete system fits together:

```
┌──────────────┐
│ Data Sources │
│ (Apps, DBs)  │
└──────┬───────┘
       ↓
┌──────────────┐
│ Pub/Sub      │  ← events: new data, drift, retrain
└──────┬───────┘
       ↓
┌──────────────┐
│ Cloud        │
│ Functions    │  ← routing + decisions
└──────┬───────┘
       ↓
┌──────────────┐
│ Vertex AI    │
│ Pipelines    │  ← preprocess → train → evaluate
└──────┬───────┘
       ↓
┌──────────────┐
│ Vertex AI    │
│ Model Reg.   │
└──────┬───────┘
       ↓
┌──────────────┐
│ Endpoints    │  ← online inference
└──────┬───────┘
       ↓
┌──────────────┐
│ Monitoring   │
│ & Drift      │
└──────┬───────┘
       ↺ (retrain loop)
```

This architecture creates a **closed-loop system** where monitoring triggers retraining, which updates models, which serve predictions, which get monitored again.

---

## 🔧 What Each Layer Does (Plain English)

### 1️⃣ Infrastructure Layer (Terraform)

**The Foundation**

This is where everything starts. All infrastructure is defined as code:

- **GCS buckets** for data storage and model artifacts
- **Pub/Sub topics** for event-driven communication
- **Service accounts & IAM** with least-privilege access
- **Cloud Functions** for orchestration
- **Vertex AI resources** for ML workloads

**Why it matters:**
- ✅ Reproducible across environments (dev, staging, prod)
- ✅ Auditable (every change is tracked in Git)
- ✅ Version-controlled (rollback is a `git revert` away)
- ✅ No manual console clicks = fewer human errors

**Example Terraform structure:**
```hcl
# Infrastructure as Code
resource "google_storage_bucket" "ml_artifacts" {
  name     = "${var.project_id}-ml-artifacts"
  location = var.region
}

resource "google_pubsub_topic" "ml_events" {
  name = "ml-events-${var.environment}"
}

resource "google_service_account" "ml_service_account" {
  account_id = "ml-sa-${var.environment}"
}
```

---

### 2️⃣ Event Layer (Pub/Sub)

**The Nervous System**

Pub/Sub decouples all components. Events flow through topics like:

- `new_data_available` → triggers data validation pipeline
- `model_drift_detected` → triggers retraining workflow
- `training_completed` → triggers model evaluation
- `evaluation_passed` → triggers deployment

**Why it matters:**
- ✅ **Loose coupling**: Services don't need to know about each other
- ✅ **Scalability**: Handle bursts without breaking
- ✅ **Resilience**: If one component fails, others continue
- ✅ **Flexibility**: Add new consumers without changing producers

**Event-driven architecture benefits:**
```
Producer → Topic → Multiple Consumers
   ↓
Cloud Function A (training)
Cloud Function B (monitoring)
Cloud Function C (alerting)
```

---

### 3️⃣ Orchestration Layer (Cloud Functions)

**The Brain**

Cloud Functions make decisions based on events:

- **Which pipeline to run?** (e.g., retrain vs. new model)
- **With what parameters?** (e.g., hyperparameters, data splits)
- **In which environment?** (dev vs. prod)

This is **business logic**, not ML code. It's the glue that connects events to ML workflows.

**Example decision logic:**
```python
def handle_ml_event(event, context):
    event_type = event['attributes']['type']
    
    if event_type == 'new_data_available':
        trigger_pipeline('data_validation')
    elif event_type == 'drift_detected':
        trigger_pipeline('retrain_model')
    elif event_type == 'training_completed':
        trigger_pipeline('evaluate_model')
```

---

### 4️⃣ ML Workflow Layer (Vertex AI Pipelines)

**The Factory**

Vertex AI Pipelines run your ML workflows as **reproducible, versioned pipelines**:

**Typical pipeline steps:**
1. **Data validation** → Check data quality, schema compliance
2. **Feature engineering** → Transform raw data into features
3. **Model training** → Train with tracked hyperparameters
4. **Evaluation** → Compute metrics (accuracy, precision, recall)
5. **Registration** → Store model in registry if metrics pass

**Why Vertex AI Pipelines:**
- ✅ **Reproducibility**: Same inputs = same outputs (guaranteed)
- ✅ **Tracking**: Every run is logged with parameters and metrics
- ✅ **Versioning**: Pipeline code is versioned, runs are tracked
- ✅ **Parallelization**: Steps run in parallel when possible
- ✅ **Cost optimization**: Only pay for compute time used

**Pipeline example (Kubeflow):**
```python
@dsl.pipeline(
    name='ml-training-pipeline',
    description='End-to-end ML training workflow'
)
def ml_pipeline(
    input_data: str,
    model_name: str
):
    validate = validate_data_op(input_data)
    features = engineer_features_op(validate.output)
    train = train_model_op(features.output)
    evaluate = evaluate_model_op(train.output)
    register = register_model_op(
        evaluate.output,
        model_name=model_name
    )
```

---

### 5️⃣ Model Management (Vertex AI Model Registry)

**The Catalog**

Every model version is stored with:

- **Model artifacts** (weights, metadata)
- **Training metrics** (accuracy, loss curves)
- **Evaluation results** (test set performance)
- **Lineage** (which data, code, and parameters produced it)
- **Tags** (production, staging, experimental)

**Why it matters:**
- ✅ **No more confusion**: "Which model is live?"
- ✅ **Easy rollbacks**: Revert to previous version in seconds
- ✅ **Compliance**: Full audit trail for regulated industries
- ✅ **Experimentation**: Compare model versions side-by-side

**Model registry workflow:**
```
Training → Evaluation → Registration → Deployment
   ↓           ↓            ↓            ↓
  v1.0       metrics      v1.0        staging
  v1.1       metrics      v1.1        production
  v2.0       metrics      v2.0        canary
```

---

### 6️⃣ Serving Layer (Vertex AI Endpoints)

**The Product**

This is what your applications actually hit for predictions:

**Features:**
- **Online inference** → Sub-100ms latency
- **Autoscaling** → Handles traffic spikes automatically
- **Canary deployments** → Gradual rollout (10% → 50% → 100%)
- **A/B testing** → Compare model versions in production
- **Traffic splitting** → Route X% to model A, Y% to model B

**Why Vertex AI Endpoints:**
- ✅ **Managed infrastructure**: No server management
- ✅ **High availability**: 99.9% uptime SLA
- ✅ **Cost-effective**: Pay per prediction
- ✅ **Security**: IAM-based access control

**Deployment strategy:**
```
New Model → Canary (10%) → Gradual Rollout → Full Production
              ↓
         Monitor metrics
              ↓
    If good: increase traffic
    If bad: rollback immediately
```

---

### 7️⃣ Monitoring & Feedback Loop

**The Immune System**

This closes the MLOps loop:

**What gets monitored:**
- **Prediction logging** → Every prediction is logged
- **Data drift detection** → Input distribution changes
- **Performance decay** → Model accuracy degrades over time
- **Latency & errors** → Serving infrastructure health

**Automatic triggers:**
- Data drift detected → Trigger retraining pipeline
- Performance below threshold → Alert + retrain
- Error rate spike → Alert + investigate

**Monitoring architecture:**
```
Predictions → Cloud Logging → Monitoring Dashboard
     ↓
Drift Detection Service
     ↓
Pub/Sub Event: "drift_detected"
     ↓
Cloud Function → Trigger Retraining
```

---

## 🔐 Security & Governance (Critical)

Security is **not optional** in production ML systems. Here's what needs to be in place:

### Service Account Strategy

- **Separate service accounts** for each component
- **Least-privilege IAM** → Only grant what's needed
- **No long-lived secrets** → Use Workload Identity

### Environment Isolation

- **Dev/Staging/Prod** → Completely separate resources
- **Network isolation** → VPCs, private endpoints
- **Data isolation** → Separate buckets, databases

### CI/CD Security

- **Approval gates** → Require reviews for production
- **Automated testing** → Validate before deployment
- **Secrets management** → Use Secret Manager, not hardcoded values

### IAM Best Practices

```hcl
# Example: Least-privilege IAM
resource "google_project_iam_member" "ml_service_account" {
  project = var.project_id
  role    = "roles/aiplatform.user"  # Only what's needed
  member  = "serviceAccount:${google_service_account.ml_service_account.email}"
}
```

**Why this matters:**
- A breach in one component doesn't compromise the entire system
- Compliance requirements (GDPR, HIPAA) are easier to meet
- Audits are straightforward (everything is in Terraform)

---

## 📊 Architecture Diagram (Detailed)

Here's a more detailed view of how components interact:

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Sources Layer                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │   Apps   │  │   DBs    │  │  APIs    │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
└───────┼─────────────┼─────────────┼─────────────────────────┘
        │             │             │
        └─────────────┴─────────────┘
                      ↓
        ┌─────────────────────────────┐
        │      Pub/Sub Topics          │
        │  ┌───────────────────────┐   │
        │  │  ml-events            │   │
        │  │  data-ingestion       │   │
        │  │  model-updates        │   │
        │  └───────────┬───────────┘   │
        └──────────────┼───────────────┘
                       ↓
        ┌─────────────────────────────┐
        │    Cloud Functions           │
        │  ┌───────────────────────┐   │
        │  │  Event Router         │   │
        │  │  Pipeline Trigger     │   │
        │  │  Decision Logic       │   │
        │  └───────────┬───────────┘   │
        └──────────────┼───────────────┘
                       ↓
        ┌─────────────────────────────┐
        │   Vertex AI Pipelines       │
        │  ┌───────────────────────┐   │
        │  │  Data Validation      │   │
        │  │  Feature Engineering │   │
        │  │  Model Training      │   │
        │  │  Evaluation          │   │
        │  └───────────┬───────────┘   │
        └──────────────┼───────────────┘
                       ↓
        ┌─────────────────────────────┐
        │  Vertex AI Model Registry   │
        │  ┌───────────────────────┐   │
        │  │  Model Versions       │   │
        │  │  Metadata & Metrics   │   │
        │  │  Lineage Tracking     │   │
        │  └───────────┬───────────┘   │
        └──────────────┼───────────────┘
                       ↓
        ┌─────────────────────────────┐
        │  Vertex AI Endpoints        │
        │  ┌───────────────────────┐   │
        │  │  Online Inference     │   │
        │  │  Autoscaling          │   │
        │  │  Canary Deployments   │   │
        │  └───────────┬───────────┘   │
        └──────────────┼───────────────┘
                       ↓
        ┌─────────────────────────────┐
        │  Monitoring & Observability │
        │  ┌───────────────────────┐   │
        │  │  Prediction Logging  │   │
        │  │  Drift Detection      │   │
        │  │  Performance Metrics  │   │
        │  └───────────┬───────────┘   │
        └──────────────┼───────────────┘
                       │
                       ↺ (feedback loop)
```
---

## 💡 Key Takeaways

1. **Infrastructure as Code is non-negotiable** → Terraform everything
2. **Event-driven architecture scales** → Pub/Sub decouples everything
3. **Reproducibility is built-in** → Vertex AI Pipelines track everything
4. **Security is layered** → Service accounts, IAM, network isolation
5. **Monitoring closes the loop** → Automated retraining based on drift

---

## 📚 Additional Resources

- [Vertex AI Pipelines Documentation](https://cloud.google.com/vertex-ai/docs/pipelines)
- [Cloud Functions Best Practices](https://cloud.google.com/functions/docs/best-practices)
- [MLOps on GCP Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)

---

## Conclusion

Building a full ML platform on GCP is complex, but it's also **incredibly valuable** for your career. This architecture gives you:

- ✅ **Production-ready patterns** you can use immediately
- ✅ **Interview talking points** that demonstrate deep understanding
- ✅ **Portfolio projects** that stand out from basic ML tutorials
- ✅ **Real-world experience** with enterprise-grade systems

The key is to **start simple** and **iterate**. Don't try to build everything at once. Pick one component, get it working, then add the next layer.

**Remember:** The best ML platform is the one that ships models to production reliably, not the one with the most features.

---

*Have questions or want to discuss this architecture? Reach out on [Hashnode](https://israelcodes.hashnode.dev) or connect on [LinkedIn](https://linkedin.com/in/Aloagbaye).*

---

**Tags:** #MachineLearning #MLOps #GCP #CloudArchitecture #Terraform #VertexAI #DevOps
