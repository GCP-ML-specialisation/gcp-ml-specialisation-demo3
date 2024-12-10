# gcp-ml-specialisation-demo3

### dataset source
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

 
To set up and run the demo, follow these steps:
 
1. **Clone the repository**.
 
2. **Update the configuration file**:
   - Fill in the values in `config_example.ini` as indicated.
   - Rename the file to `config.ini` after populating the required values.
 
3. **Set up Google Cloud configuration**:
   - Set your project ID:
     ```bash
     PROJECT_ID="<your_project_id>"
     gcloud config set project ${PROJECT_ID}
     ```
   - Set your account ID:
     ```bash
     ACCOUNT_ID="<your_account_id>"
     gcloud config set account ${ACCOUNT_ID}
     ```
   - Create a new configuration:
     ```bash
     CONFIG="<your_config_name>"
     gcloud config configurations create ${CONFIG}
     ```
   - Authenticate with Google Cloud:
     ```bash
     gcloud auth application-default login
     gcloud auth login
     ```
 
4. **Set up the service account**:
   - Use the following service account:
     ```bash
     SERVICE_ACCOUNT="<your service account>"
     ```
 
5. **Optional (only if code changes are needed)**:
   - Run `ml_spec3_pipeline.py` to generate the necessary YAML files for the pipeline components.
 
6. **Run the pipeline**:
   - Execute the `run_pipeline.py` file to initiate the pipeline.