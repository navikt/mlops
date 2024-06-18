### mlops


principalSet://iam.googleapis.com/projects/368669329223/locations/global/workloadIdentityPools/github-actions-cloud-run/attribute.repository/navikt/mlops

POOL ID = github-actions-cloud-run
PROVIDER-ID = github
PROJECT= 368669329223

serivce account: team-tiltak-service-account@team-tiltak-dev-2137.iam.gserviceaccount.com

WIF-workload_identity_provider: projects/{project}/locations/global/workloadIdentityPools/{pool-id}/providers/{provider-id}
WIF : projects/368669329223/locations/global/workloadIdentityPools/github-actions-cloud-run/providers/github



https://stackoverflow.com/questions/75840164/permission-artifactregistry-repositories-uploadartifacts-denied-on-resource-usin
gcloud auth login
gcloud auth configure-docker europe-docker.pkg.dev

https://github.com/navikt/knada-images/blob/main/.github/workflows/airflow.yaml