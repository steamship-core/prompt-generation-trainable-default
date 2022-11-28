import csv
import json
import logging
import os
import time
from pathlib import Path
from unittest.mock import patch

import requests
from requests_file import FileAdapter
from steamship import Block, File, Tag, Task, TaskState
from steamship.invocable import InvocableResponse
from steamship.plugin.inputs.block_and_tag_plugin_input import \
    BlockAndTagPluginInput
from steamship.plugin.inputs.train_plugin_input import TrainPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import \
    BlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest

from src.api import GCPVertexAITrainableTaggerPlugin
from src.model import ThirdPartyTrainingStatus, VertexAIModel

__copyright__ = "Steamship"
__license__ = "MIT"

# If this isn't present, Localstack won't show logs
logging.getLogger().setLevel(logging.INFO)

def _load_config() -> dict:
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'test_config.json'), 'r') as config_file:
        config = json.load(config_file)
    return config

def _load_vertex_model() -> VertexAIModel:
    _temp_tagger = GCPVertexAITrainableTaggerPlugin(config=_load_config())
    model_folder = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test_data'))
    model = VertexAIModel()
    model.receive_config(_temp_tagger.config)
    model.load_from_folder(model_folder)
    return model

def test_init():
    model = _load_vertex_model()

def test_run():
    model = _load_vertex_model()
    assert model.endpoint_path is not None

    tagger = GCPVertexAITrainableTaggerPlugin(config=_load_config())
    test_request = PluginRequest(data=BlockAndTagPluginInput(file=File(blocks=[Block(text="I got a high score on my math final!"), Block(text="I like to camp on the weekends")])))
    output : InvocableResponse[BlockAndTagPluginOutput] = tagger.run_with_model( test_request, model)

    file = output.data['file']
    assert file is not None
    assert len(file['blocks']) == 2
    assert len(file['blocks'][0]['tags']) > 0
    assert len(file['blocks'][1]['tags']) > 0



def test_train():
    logging.root.setLevel(logging.INFO)

    # So that requests can fetch the test training file from a file:// url
    requests_session = requests.session()
    requests_session.mount('file://', FileAdapter())

    model_folder = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test_data'))
    test_data_path_url = (model_folder / 'test_training_data.jsonl').as_uri()

    model = _load_vertex_model()
    tpi = TrainPluginInput(plugin_instance='None for test', training_data_url=test_data_path_url)
    tpo = model.train(tpi)
    assert tpo is not None
    assert tpo.status.state == TaskState.running
    assert tpo.status.remote_status_input is not None
    assert tpo.status.remote_status_input.get('dataset_start_time') is not None
    assert tpo.status.remote_status_input.get('training_job_name') is not None
    assert tpo.status.remote_status_input.get('training_started') is False

    assert tpo.status.remote_status_output is not None
    assert tpo.status.remote_status_output.get("state") == ThirdPartyTrainingStatus.PREPARING_DATASET.value

    assert model._is_awaiting_dataset_deployment(tpo.status.remote_status_input.get('dataset_start_time')) is True

    # Check status should await dataset deployment
    tso = model.train_status(PluginRequest(status=Task(
        state=TaskState.running,
        remote_status_input=tpo.status.remote_status_input
    ), data=tpi))
    assert tso is not None
    assert tso.status.remote_status_output is not None
    assert tso.status.remote_status_output.get("state") == ThirdPartyTrainingStatus.PREPARING_DATASET.value

def test_data_done_start_model_deploy():
    model_display_name = "FOO"

    # Now we'll mock that the dataset is done.
    def _is_awaiting_dataset_deployment(self, endpoint):
        return False

    def _start_training(self, training_job_name: str, multi_class: bool):
        return model_display_name

    tpo = InvocableResponse(
        status=Task(
            status=TaskState.running,
            remote_status_input={
                "dataset_start_time": time.time(),
                "training_job_name": "foo",
                "training_started": False
            }
        )
    )
    tspi = PluginRequest(
        status=tpo.status
    )

    with patch.object(VertexAIModel, '_is_awaiting_dataset_deployment', _is_awaiting_dataset_deployment):
        with patch.object(VertexAIModel, '_start_training', _start_training):
            # Test that the patch worked.
            model = _load_vertex_model()
            assert model._is_awaiting_dataset_deployment(tpo.status.remote_status_input.get('dataset_start_time')) is False
            tso = model.train_status(tspi)
            assert tso is not None
            assert tso.status is not None
            assert tso.status.remote_status_output is not None
            assert tso.status.remote_status_output.get("state") == ThirdPartyTrainingStatus.TRAINING.value
            assert tso.status.remote_status_output.get("model_display_name") == model_display_name
            assert tso.status.remote_status_output.get("training_started") is True
            assert tso.status.state == TaskState.running


def test_training_in_progress():
    model_display_name = "FOO"

    # Now we'll mock that the dataset is done.
    def _is_awaiting_dataset_deployment(self, endpoint):
        return False

    def _start_training(self, training_job_name: str, multi_class: bool):
        return model_display_name

    def _training_job_complete(self, name):
        return False

    tpo = InvocableResponse(
        status=Task(
            state=TaskState.running,
            remote_status_output=dict(
                model_display_name=model_display_name
            ),
            remote_status_input=dict(
                dataset_start_time=time.time(),
                training_job_name="foo",
                training_started=True  # Note -- this is what signals that we're already training!
            )
        )
    )
    tspi = PluginRequest(
        status=tpo.status
    )

    with patch.object(VertexAIModel, '_is_awaiting_dataset_deployment', _is_awaiting_dataset_deployment):
        with patch.object(VertexAIModel, '_start_training', _start_training):
            with patch.object(VertexAIModel, '_training_job_complete', _training_job_complete):
                # Test that the patch worked.
                model = _load_vertex_model()
                assert model._is_awaiting_dataset_deployment(tpo.status.remote_status_input.get('dataset_start_time')) is False
                tso = model.train_status(tspi)
                message = 'Job not complete - continuing to wait.' # Note -- this is how we know we haven't doubly kicked off training a second time by accident!
                assert tso is not None
                assert tso.status.remote_status_input is not None
                assert tso.status.remote_status_output.get("state") == ThirdPartyTrainingStatus.TRAINING.value
                assert tso.status.remote_status_output.get("message") is not None
                assert message in tso.status.remote_status_output.get("message")
                assert tso.status.remote_status_input.get("training_started") is True
                assert tso.status.state == TaskState.running


def test_endpoint_deploy_init():
    model_display_name = "FOO"

    # Now we'll mock that the dataset is done.
    def _is_awaiting_dataset_deployment(self, endpoint):
        return False

    def _training_job_complete(self, name):
        return True # This will cause us to proceed to endpoint deployment

    def _get_model_handle(self, name):
        return {}

    def _get_endpoint_handle(self, name):
        return None # This is what indicates the endpoint is not yet deployed.

    def _deploy_endpoint(self, name, model):
        return None

    tpo = InvocableResponse(
        status=Task(
            state=TaskState.running,
            remote_status_output=dict(
                model_display_name=model_display_name
            ),
            remote_status_input=dict(
                dataset_start_time=time.time(),
                training_job_name="foo",
                training_started=True # Note -- this is what signals that we're already training!
            )
        )
    )
    tspi = PluginRequest(
        status=tpo.status
    )

    with patch.object(VertexAIModel, '_is_awaiting_dataset_deployment', _is_awaiting_dataset_deployment):
        with patch.object(VertexAIModel, '_training_job_complete', _training_job_complete):
            with patch.object(VertexAIModel, '_get_model_handle', _get_model_handle):
                with patch.object(VertexAIModel, '_get_endpoint_handle', _get_endpoint_handle):
                    with patch.object(VertexAIModel, '_deploy_endpoint', _deploy_endpoint):
                        # Test that the patch worked.
                        model = _load_vertex_model()
                        assert model._is_awaiting_dataset_deployment(tpo.status.remote_status_input.get('dataset_start_time')) is False
                        tso = model.train_status(tspi)
                        assert tso is not None
                        assert tso.status.remote_status_output is not None
                        assert tso.status.remote_status_output.get("state") == ThirdPartyTrainingStatus.DEPLOYING_ENDPOINT.value
                        assert tso.status.remote_status_output.get("message") is not None
                        assert tso.status.remote_status_input.get("training_started") is True
                        assert tso.status.state is TaskState.running

def test_endpoint_deploy_in_progress():
    model_display_name = "FOO"

    # Now we'll mock that the dataset is done.
    def _is_awaiting_dataset_deployment(self, endpoint):
        return False

    def _training_job_complete(self, name):
        return True # This will cause us to proceed to endpoint deployment

    def _get_model_handle(self, name):
        return {}

    endpoint_resource_name = "mock_resource_name"

    def _get_endpoint_handle(self, name):
        class Endpoint:
            resource_name: str = endpoint_resource_name
        return Endpoint()

    def _is_endpoint_deploy_complete(self, endpoint):
        return False

    tpo = InvocableResponse(
        status=Task(
            state=TaskState.running,
            remote_status_output=dict(
                model_display_name=model_display_name
            ),
            remote_status_input=dict(
                dataset_start_time=time.time(),
                training_job_name="foo",
                training_started=True # Note -- this is what signals that we're already training!
            )
        )
    )
    tspi = PluginRequest(
        status=tpo.status
    )

    with patch.object(VertexAIModel, '_is_awaiting_dataset_deployment', _is_awaiting_dataset_deployment):
        with patch.object(VertexAIModel, '_training_job_complete', _training_job_complete):
            with patch.object(VertexAIModel, '_get_model_handle', _get_model_handle):
                with patch.object(VertexAIModel, '_get_endpoint_handle', _get_endpoint_handle):
                    with patch.object(VertexAIModel, '_is_endpoint_deploy_complete', _is_endpoint_deploy_complete):
                        # Test that the patch worked.
                        model = _load_vertex_model()
                        assert model._is_awaiting_dataset_deployment(tpo.status.remote_status_input.get('dataset_start_time')) is False
                        tso = model.train_status(tspi)
                        assert tso is not None
                        assert tso.status.remote_status_output is not None
                        assert tso.status.remote_status_output.get("state") == ThirdPartyTrainingStatus.AWAITING_ENDPOINT.value
                        assert tso.status.remote_status_output.get("message") is not None
                        assert tso.status.remote_status_input.get("training_started") is True
                        assert tso.status.remote_status_output.get("endpoint_resource_name") == endpoint_resource_name
                        assert tso.status.state is TaskState.running



def test_endpoint_deploy_complete():
    model_display_name = "FOO"

    # Now we'll mock that the dataset is done.
    def _is_awaiting_dataset_deployment(self, endpoint):
        return False

    def _training_job_complete(self, name):
        return True # This will cause us to proceed to endpoint deployment

    def _get_model_handle(self, name):
        return {}

    endpoint_resource_name = "mock_resource_name"

    def _get_endpoint_handle(self, name):
        class Endpoint:
            resource_name: str = endpoint_resource_name
        return Endpoint()

    def _is_endpoint_deploy_complete(self, endpoint):
        return True # This indicates the endpoint deployment is complete.

    tpo = InvocableResponse(
        status=Task(
            state=TaskState.running,
            remote_status_output=dict(
                model_display_name=model_display_name
            ),
            remote_status_input=dict(
                dataset_start_time=time.time(),
                training_job_name="foo",
                training_started=True # Note -- this is what signals that we're already training!
            )
        )
    )
    tspi = PluginRequest(
        status=tpo.status
    )

    with patch.object(VertexAIModel, '_is_awaiting_dataset_deployment', _is_awaiting_dataset_deployment):
        with patch.object(VertexAIModel, '_training_job_complete', _training_job_complete):
            with patch.object(VertexAIModel, '_get_model_handle', _get_model_handle):
                with patch.object(VertexAIModel, '_get_endpoint_handle', _get_endpoint_handle):
                    with patch.object(VertexAIModel, '_is_endpoint_deploy_complete', _is_endpoint_deploy_complete):
                        # We'll intentionally not patch the model evaluation to see if the error handling works.
                        # Test that the patch worked.
                        model = _load_vertex_model()
                        assert model._is_awaiting_dataset_deployment(tpo.status.remote_status_input.get('dataset_start_time')) is False
                        tso = model.train_status(tspi)
                        assert tso is not None
                        assert tso.status.remote_status_input is not None
                        assert tso.status.remote_status_output.get("state") == ThirdPartyTrainingStatus.TRAINED.value
                        assert tso.status.remote_status_output.get("message") is not None
                        assert tso.status.remote_status_output.get("training_started") is True
                        assert tso.status.remote_status_output.get("endpoint_resource_name") == endpoint_resource_name
                        # The output has a duplicate at the very end.
                        assert tso.status.output.get("state") == ThirdPartyTrainingStatus.TRAINED.value
                        assert tso.status.output.get("message") is not None
                        assert tso.status.output .get("training_started") is True
                        assert tso.status.output.get("endpoint_resource_name") == endpoint_resource_name
                        assert tso.status.state is TaskState.succeeded



# def test_MANUAL_fetch_model_evaluation():
#     """Note:
#
#     This test isn't intended to be run automatically. It needs to be pre-configured
#     with credentials and an existing trained model to succeed. It's goal is for an offline
#     sanity check of our interpretation of Google's results.
#     """
#     model_display_name = "model-training-task-0BEF05C8-C240-40C8-B681-566B982CB5E5"
#     model = _load_vertex_model()
#     model._prepare_credentials()
#     handle = model._get_model_handle(model_display_name)
#     evaluation = model._fetch_model_evaluation(handle)
#     print(evaluation)
#     assert isinstance(evaluation, dict)
#
#
# def test_MANUAL_deploy_model_endpoint():
#     """Note:
#
#     This test isn't intended to be run automatically. It needs to be pre-configured
#     with credentials and an existing trained model to succeed. It's goal is for an offline
#     sanity check of our interpretation of Google's results.
#
#     It's a long running test!
#
#     Precondition: the model should be DEPLOYED but have no ENDPOINT
#     """
#
#     training_job_name = "training-task-40E0B624-0419-4E4A-B4CC-FA1E4F971F70"
#     model_display_name = f"model-{training_job_name}"
#
#     tpo = TrainPluginOutput(
#         training_complete=False,
#         training_results=dict(
#             model_display_name=model_display_name
#         ),
#         training_reference_data=dict(
#             dataset_start_time=time.time() - 1000,
#             training_job_name=training_job_name,
#             training_started=True # Note -- this is what signals that we're already training!
#         )
#     )
#     tspi = TrainStatusPluginInput(
#         training_reference_data=tpo.training_reference_data
#     )
#
#     model = _load_vertex_model()
#     model._prepare_credentials()
#     model.endpoint_path = None
#
#     # Call status, triggering endpoint deployment.
#     tso = model.train_status(tspi)
#     assert tso is not None
#     assert tso.training_results is not None
#     assert tso.training_results.get("state") == ThirdPartyTrainingStatus.DEPLOYING_ENDPOINT.value
#
#     # Wait
#     time.sleep(3)
#
#     # Call status, triggering endpoint deployment.
#     tso = model.train_status(tspi)
#     assert tso is not None
#     assert tso.training_results is not None
#     assert tso.training_results.get("state") == ThirdPartyTrainingStatus.AWAITING_ENDPOINT.value
#
#     # Wait 15 minutes
#     time.sleep(15 * 60)
#
#     # Call status, triggering endpoint deployment.
#     tso = model.train_status(tspi)
#     assert tso is not None
#     assert tso.training_results is not None
#     assert tso.training_results.get("state") == ThirdPartyTrainingStatus.TRAINED.value
#     assert tso.training_complete is True
#     assert tso.training_results.get("endpoint_resource_name") is not None


# Test
def test_prepare_spam_training_data():
    folder = os.path.dirname(os.path.abspath(__file__))
    files = []
    with open(os.path.join(folder, '..', 'test_data', 'spam-data.csv'), newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            if row[0] != 'Category':
                files.append(File(blocks=[
                    Block(text=row[1],
                          tags=[
                              Tag(kind='classification', name=row[0])
                          ]
                          )
                ]))
    with open(os.path.join(folder, '..', 'test_data', 'test_training_data.jsonl'), 'w') as training_data_file:
        for file in files:
            training_data_file.write(json.dumps(file.dict()))
            training_data_file.write('\n')
