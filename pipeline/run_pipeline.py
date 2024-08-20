import google.cloud.aiplatform as aip
import configparser
config = configparser.ConfigParser()
config.read("../config.ini")




aip.init(
    project=config['gcp_pipeline']['PROJECT_ID'],
    location=config['gcp']['region'],
    staging_bucket=config['gcp_pipeline']['BUCKET_URI'],
    service_account=config['gcp_pipeline']['SERVICE_ACCOUNT'],
)

job = aip.PipelineJob(
    display_name=config['gcp_pipeline']['DISPLAY_NAME'],
    enable_caching=True,
    template_path=config['gcp_pipeline']['TEMPLATE_PATH'],
    pipeline_root=f"{config['gcp_pipeline']['BUCKET_URI']}pipeline-root/",
)

job.run()
