from kfp import compiler
import google.cloud.aiplatform as aip
from kfp import dsl
from kfp.dsl import ConcatPlaceholder
from pre_processing_modules import (
    basic_preprocessing,
    data_transformation,
)
from model import (
    create_dataset,
    training,

)
import configparser
config = configparser.ConfigParser()
config.read("../config.ini")


aip.init(project=config['gcp_pipeline']['PROJECT_ID'], staging_bucket=config['gcp_pipeline']['BUCKET_URI'], service_account=SERVICE_ACCOUNT)


@dsl.pipeline(
    name="intro-pipeline-unique",
    description="A simple intro pipeline",
    pipeline_root=config['gcp_pipeline']['PIPELINE_ROOT'].format(config['gcp_pipeline']['BUCKET_URI']),
)
def pipeline(
    BUCKET_URI: str = config['gcp_pipeline']['BUCKET_URI'],
    FILE: str = config['gcp_pipeline']['FILE'],
):

    pre_processed_df = basic_preprocessing(
        BUCKET_URI=BUCKET_URI,
        FILE=FILE
    )
    feature_engineered_df = data_transformation(
        df=pre_processed_df.output
    )
   
    # create_automl_dataset = create_dataset(
    #     processed_ds= feature_engineered_df.output,
    #     project_id=PROJECT_ID
    # )
    #print(create_automl_dataset.output['dataset'])
    model = training(processed_ds=feature_engineered_df.output, project_id=PROJECT_ID)

    # model_evaluation(
    #     test_set=ready_dataset.outputs["dataset_valid"],
    #     training_model=model.outputs["trained_model"],
    # )

    # deploy_xgboost_model(
    #     model=model.outputs["trained_model"],
    #     project_id=PROJECT_ID,
    # )


compiler.Compiler().compile(pipeline_func=pipeline, package_path="mlspec3_pipeline.json")
