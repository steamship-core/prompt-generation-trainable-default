"""Configuration for the GCP Model."""
from typing import Optional

from steamship.invocable import Config


class GCPConfig(Config):
    project_id: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    auth_uri: str
    token_uri: str
    auth_provider_x509_cert_url: str
    client_x509_cert_url: str
    region: str
    training_data_bucket: str
    single_or_multi_label: str
    tag_kind: str
    include_tag_names: Optional[str]
