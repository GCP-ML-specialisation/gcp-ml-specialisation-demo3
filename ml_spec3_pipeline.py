from kfp import compiler
import google.cloud.aiplatform as aip
from kfp import dsl
from kfp.dsl import ConcatPlaceholder
from pre_processing_modules import (
    basic_preprocessing,
    data_transformation,
)
from model import (
    training,
    deploy_automl_model

)
import configparser
def load_config():
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config

config = load_config()


aip.init(project=config['gcp_pipeline']['PROJECT_ID'],
staging_bucket=config['gcp_pipeline']['BUCKET_URI'],
service_account=config['gcp_pipeline']['SERVICE_ACCOUNT'])


@dsl.pipeline(
    name="intro-pipeline-unique",
    description="A simple intro pipeline",
    pipeline_root=config['gcp_pipeline']['PIPELINE_ROOT'].format(config['gcp_pipeline']['BUCKET_URI']),
)
def pipeline(
    BUCKET_URI: str = config['gcp_pipeline']['BUCKET_URI'],
    FILE1: str = config['gcp_pipeline']['FILE1'],
    FILE2: str = config['gcp_pipeline']['FILE2'],
):

    pre_processed_df = basic_preprocessing(
        BUCKET_URI=BUCKET_URI,
        FILE1=FILE1,
        FILE2=FILE2
    )
    feature_engineered_df = data_transformation(
        df=pre_processed_df.output,
        BUCKET_URI=BUCKET_URI
    )
   
    model = training(processed_ds=feature_engineered_df.output, project_id=config['gcp_pipeline']['PROJECT_ID'], region=config['gcp']['region'])
    

    deploy_automl_model(
        model_artifact=model.outputs['train_model'],
        project_id=config['gcp_pipeline']['PROJECT_ID'],
    )


compiler.Compiler().compile(pipeline_func=pipeline, package_path="output_files/mlspec3_pipeline.json")
