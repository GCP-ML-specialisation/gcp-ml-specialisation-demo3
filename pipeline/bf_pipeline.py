from kfp import compiler
import google.cloud.aiplatform as aip
from kfp import dsl
from kfp.dsl import ConcatPlaceholder
from pre_processing_modules import (
    basic_preprocessing,
    data_transformation,
    train_validation_test_split,
)

from model import model_evaluation, training, deploy_xgboost_model


PROJECT_ID = "prj-dev-mlbf-flt-01-29e5" # replace 
BUCKET_URI = "gs://pipeline_black_friday/" # replace 
SERVICE_ACCOUNT = (
    "project-service-account@prj-dev-mlbf-flt-01-29e5.iam.gserviceaccount.com"
) # replace 
PIPELINE_ROOT = "{}/pipeline_root".format(BUCKET_URI) # replace 

aip.init(project=PROJECT_ID, staging_bucket=BUCKET_URI, service_account=SERVICE_ACCOUNT)


@dsl.pipeline(
    name="intro-pipeline-unique",
    description="A simple intro pipeline",
    pipeline_root=PIPELINE_ROOT,
)
def pipeline(
    train_name: str = "train.csv",
    test_name: str = "test.csv",
    BUCKET_URI: str = BUCKET_URI,
    raw_folder: str = "raw_data/",
):

    pre_processed_df = basic_preprocessing(
        bucket_URI=BUCKET_URI,
        folder=raw_folder,
        train=train_name,
        test=test_name,
    )
    feature_engineered_df = data_transformation(
        df_train=pre_processed_df.outputs["dataset_train"],
        df_test=pre_processed_df.outputs["dataset_test"],
    )
    ready_dataset = train_validation_test_split(
        df_train=feature_engineered_df.outputs["dataset_train"]
    )

    model = training(df_train=ready_dataset.outputs["dataset_train"])

    model_evaluation(
        test_set=ready_dataset.outputs["dataset_valid"],
        training_model=model.outputs["trained_model"],
    )

    deploy_xgboost_model(
        model=model.outputs["trained_model"],
        project_id=PROJECT_ID,
    )


compiler.Compiler().compile(pipeline_func=pipeline, package_path="bf_pipeline.json")
