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
    train_model: Output[Model],
):

    from google.cloud import aiplatform
    
    aiplatform.init(project=project_id)
    
    gcs_uri = processed_ds.uri + ".csv"

    
    dataset = aiplatform.TabularDataset.create(
        display_name="HR Analytics3",
        gcs_source=gcs_uri,
        )
 
    
    label_column = "Attrition"
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name="train-automl-hr-analytics1",
        optimization_prediction_type="classification",
        optimization_objective="maximize-au-prc",
        )

    model = job.run(
        dataset=dataset,
        target_column=label_column,
        training_fraction_split=0.6,
        validation_fraction_split=0.2,
        test_fraction_split=0.2,
        budget_milli_node_hours=1000,
        model_display_name="test1",
        disable_early_stopping=False,
        )
    
    train_model.uri = model.resource_name
    



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

    endpoint = model.deploy(machine_type='n1-standard-4')
    vertex_endpoint.uri = endpoint.resource_name
    vertex_model.uri = model.resource_name
