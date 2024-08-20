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
    output_component_file="create_dataset.yaml",
    base_image="python:3.11",
)
def create_dataset(processed_ds: Input[Dataset],
                   project_id: str,
                   dataset: Output[Dataset]):
    
    from google.cloud import aiplatform
    
    aiplatform.init(project=project_id)
    gcs_uri = processed_ds.uri + ".csv"

    
    dataset = aiplatform.TabularDataset.create(
        display_name="HR Analytics3",
        gcs_source=gcs_uri,
        )
    
    
    

@component(
    packages_to_install=["google-cloud-aiplatform"],
    output_component_file="training.yaml",
    base_image="python:3.11",
)
def training(processed_ds: Input[Dataset],
             project_id: str,
             model: Output[Model]):

    from google.cloud import aiplatform
    
    aiplatform.init(project=project_id)
    
    gcs_uri = processed_ds.uri + ".csv"

    
    dataset = aiplatform.TabularDataset.create(
        display_name="HR Analytics3",
        gcs_source=gcs_uri,
        )
    # print(f'This is the dataset {dataset.outputs['dataset']}')
    
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


@component(
    packages_to_install=["pandas", "gcsfs", "scikit-learn", "xgboost", "joblib"],
    output_component_file="model_evaluation.yaml",
    base_image="python:3.11",
)
def model_evaluation(
    test_set: Input[Dataset],
    training_model: Input[Model],
    kpi: Output[Metrics],
):
    pass


@component(
    packages_to_install=["google-cloud-aiplatform==1.25.0"],
)
def deploy_automl_model(
    model: Input[Model],
    project_id: str,
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model],
):
    """Deploys an XGBoost model to Vertex AI Endpoint.

    Args:
        model: The model to deploy.
        project_id: The project ID of the Vertex AI Endpoint.

    Returns:
        vertex_endpoint: The deployed Vertex AI Endpoint.
        vertex_model: The deployed Vertex AI Model.
    """
    from google.cloud import aiplatform

    aiplatform.init(project=project_id)

    endpoint = model.deploy(deployed_model_display_name='test1',
    machine_type='n1-standard-4')
