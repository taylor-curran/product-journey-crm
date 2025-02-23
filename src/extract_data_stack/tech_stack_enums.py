from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Literal


class CloudProvider(str, Enum):
    AWS = "AWS"
    AZURE = "Azure"
    GCP = "GCP"
    OCI = "OCI"
    ON_PREM = "On-Prem"


class OrchestrationTool(str, Enum):
    DAGSTER = "Dagster"
    HOME_GROWN_ADVANCED = "Home-Grown Advanced Orchestration Tool"
    HOME_GROWN_BASIC = "Home-Grown Basic Orchestration Tool"
    AIRFLOW_MWAA_AWS_MANAGED = "Airflow (MWAA) AWS Managed"
    AIRFLOW_ASTRONOMER = "Airflow (Astronomer)"
    AIRFLOW_AZURE = "Airflow (Azure Managed)"
    AIRFLOW_GCP_CLOUD_COMPOSER = "Airflow (GCP Managed) Cloud Composer"
    AIRFLOW_OSS_ON_PREM = "Airflow (OSS) On-Prem"
    AIRFLOW_NOT_SPECIFIED = "Airflow (Not Specified)"
    ACTIVEBATCH = "ActiveBatch"
    TEMPORAL = "Temporal"
    CONTROL_M = "Control-M (BMC)"
    INFORMATICA = "Informatica PowerCenter"
    ALTERYX = "Alteryx"
    SQL_SERVER_JOBS = "SQL Server Jobs"
    AWS_STEP_FUNCTIONS = "AWS Step Functions"
    AWS_LAMBDA_FUNCTIONS = "AWS Lambda Functions"
    AZURE_FUNCTIONS = "Azure Functions"
    AZURE_DATA_FACTORY = "Azure Data Factory"
    GCP_CLOUD_RUN = "GCP Cloud Run"
    GCP_CLOUD_SCHEDULER = "GCP Cloud Scheduler"
    GCP_CLOUD_FUNCTIONS = "GCP Cloud Functions"
    IBM_WORKLOAD_SCHEDULER = "IBM Workload Scheduler"
    MATILLION = "Matillion"
    AUTOSYS = "AutoSys"
    TALEND = "Talend"
    DATASTAGE = "DataStage (IBM)"
    SSIS = "SQL Server Integration Services (SSIS)"
    BOOMI = "Boomi"
    SNAPLOGIC = "SnapLogic"
    MULESOFT = "MuleSoft"
    OTHER_LEGACY_SYSTEM = "Other Legacy System"
    CAMUNDA = "Camunda"
    OTHER = "Other"
