import csv
import json
import logging
import os
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import requests
from requests_file import FileAdapter
from steamship import Block, File, Tag, Task, TaskState
from steamship.data import TagValue
from steamship.invocable import InvocableResponse
from steamship.plugin.inputs.block_and_tag_plugin_input import \
    BlockAndTagPluginInput
from steamship.plugin.inputs.train_plugin_input import TrainPluginInput
from steamship.plugin.outputs.block_and_tag_plugin_output import \
    BlockAndTagPluginOutput
from steamship.plugin.request import PluginRequest

from src.api import PromptGenerationTrainablePlugin
from src.model import ThirdPartyTrainingStatus, OpenAIModel

__copyright__ = "Steamship"
__license__ = "MIT"

# If this isn't present, Localstack won't show logs
logging.getLogger().setLevel(logging.INFO)

def _load_config() -> dict:
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'test_config.json'), 'r') as config_file:
        config = json.load(config_file)
    return config

def _load_vertex_model() -> OpenAIModel:
    _temp_tagger = PromptGenerationTrainablePlugin(config=_load_config())
    model_folder = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'test_data'))
    model = OpenAIModel()
    model.receive_config(_temp_tagger.config)
    model.load_from_folder(model_folder)
    model.training_task_id = "tst" + str(uuid.uuid4())
    return model

def test_init():
    _ = _load_vertex_model()

def test_run():
    model = _load_vertex_model()
    assert model.fine_tuned_model_name is not None

    tagger = PromptGenerationTrainablePlugin(config=_load_config())
    test_request = PluginRequest(data=BlockAndTagPluginInput(file=File(blocks=[Block(text="Gimme an X!"), Block(text="Gimme a Y!")])))
    output : InvocableResponse[BlockAndTagPluginOutput] = tagger.run_with_model( test_request, model)

    file = output.data.file
    assert file is not None
    assert len(file.blocks) == 2
    assert len(file.blocks[0].tags) == 1
    assert "X!" in file.blocks[0].tags[0].value.get(TagValue.STRING_VALUE.value)
    assert len(file.blocks[1].tags) == 1
    assert "Y!" in file.blocks[1].tags[0].value.get(TagValue.STRING_VALUE.value)


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
    assert tpo.status.remote_status_input.get('training_job_id') is not None

    assert tpo.status.remote_status_output is not None
    assert tpo.status.remote_status_output.get("state") == ThirdPartyTrainingStatus.TRAINING.value

    # Check status should await dataset deployment
    tso = model.train_status(PluginRequest(status=Task(
        state=TaskState.running,
        remote_status_input=tpo.status.remote_status_input
    ), data=tpi))
    assert tso is not None
    assert tso.status.remote_status_output is not None
    assert tso.status.remote_status_output.get("state") == ThirdPartyTrainingStatus.TRAINING.value

    # wait 30 sec; it should still be running
    time.sleep(30)
    tso = model.train_status(PluginRequest(status=Task(
        state=TaskState.running,
        remote_status_input=tpo.status.remote_status_input
    ), data=tpi))
    assert tso is not None
    assert tso.status.remote_status_output is not None
    assert tso.status.remote_status_output.get("state") == ThirdPartyTrainingStatus.TRAINING.value


    # Wait four minutes; it should be done
    time.sleep(240)
    tso = model.train_status(PluginRequest(status=Task(
        state=TaskState.running,
        remote_status_input=tpo.status.remote_status_input
    ), data=tpi))
    assert tso is not None
    assert tso.status.remote_status_output is not None
    assert tso.status.remote_status_output.get("state") == ThirdPartyTrainingStatus.TRAINED.value