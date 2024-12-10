from kfp.v2.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model,
    Artifact,
    Metrics,
)

import configparser
config = configparser.ConfigParser()
config.read("../config.ini")

    
@component(
    packages_to_install=["google-cloud-aiplatform"],
    output_component_file="output_files/training.yaml",
    base_image="python:3.11",
)
def training(
    processed_ds: Input[Dataset],
    project_id: str,
    region: str,
    train_model: Output[Model],
):

    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region)
    gcs_uri = processed_ds.uri + ".csv"
    
    
    dataset = aiplatform.ImageDataset.create(
        display_name="multi_class_image_dataset",
        gcs_source=[gcs_uri],
        import_schema_uri=aiplatform.schema.dataset.ioformat.image.single_label_classification,
    )
    model = aiplatform.AutoMLImageTrainingJob(
        display_name="image_classification_training",
        prediction_type="classification",  # Use "classification" for multi-class classification
        multi_label=False,  # Set to True if it's a multi-label classification problem
    )
    model_job = model.run(
        dataset=dataset,
        model_display_name="image_classification_model",
        budget_milli_node_hours=8000,  # Budget in milli node hours (8,000 = 8 node hours)
        disable_early_stopping=False,  # Early stopping for efficiency
    )
    
    train_model.metadata["resourceName"] = model_job.resource_name
    train_model.uri = model_job.resource_name
    



@component(
    packages_to_install=["google-cloud-aiplatform==1.25.0"],
    output_component_file="output_files/deploy_automl_model.yaml"
)

def deploy_automl_model(
    model_artifact: Input[Artifact],  # Adjusted input type
    project_id: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    """Deploys an AutoML model to Vertex AI Endpoint.

    Args:
        model_artifact: The artifact containing the trained model.
        project_id: The project ID of the Vertex AI Endpoint.

    Returns:
        vertex_endpoint: The deployed Vertex AI Endpoint.
        vertex_model: The deployed Vertex AI Model.
    """
    from google.cloud import aiplatform
    aiplatform.init(project=project_id)

    model = aiplatform.Model(model_name=model_artifact.uri)

    
    endpoint = model.deploy(
    machine_type="n1-standard-4",  # Adjust machine type as needed
    min_replica_count=1,          # Minimum number of replicas
    max_replica_count=1,          # Maximum number of replicas must match minimum for this model type
    )

    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = model.resource_name
