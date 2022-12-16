import json
import logging
import os
import uuid

from steamship import Block, File, Tag
from steamship.client import Steamship
from steamship.data import TagValue
from steamship.plugin.inputs.export_plugin_input import ExportPluginInput
from steamship.plugin.inputs.training_parameter_plugin_input import \
    TrainingParameterPluginInput

PLUGIN_HANDLE = "prompt-generation-trainable-default"
EXPORTER_HANDLE = "signed-url-exporter-1.0"
KEYWORDS = ["product", "coupon"]

def create_example_blocks() -> [Block.CreateRequest]:
    blocks = []
    for letter in "STEAMSHIP":
        block = Block.CreateRequest(text=f'Gimme a {letter}!')
        block.tags = [Tag(kind="training_generation", startIdx=0, endIdx=len(block.text),
                          value={TagValue.STRING_VALUE.value: letter})]
        blocks.append(block)
    return blocks


def test_e2e_trainable_tagger_lambda_training():
    client = Steamship()

    client.switch_workspace(workspace_handle=str(uuid.uuid4()))
    logging.info(f"Workspace id: {client.get_workspace().id}")
    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'test_config.json'), 'r') as config_file:
        config = json.load(config_file)

    config['num_completions'] = 2

    trainable_tagger = client.use_plugin(
        plugin_handle=PLUGIN_HANDLE,
        version='0.0.13',
        config=config,
    )


    file = File.create(client, blocks=create_example_blocks())

    # Now train the plugin
    training_request = TrainingParameterPluginInput(
        plugin_instance=trainable_tagger.handle,
        export_plugin_input=ExportPluginInput(
            plugin_instance=EXPORTER_HANDLE, type="file", query="blocktag",
        ),
        training_epochs = 2,
        training_params={
            "model":"babbage"
        },
    )

    train_result = trainable_tagger.train(training_request)

    train_result.wait(max_timeout_s=500, retry_delay_s=10)

    test_file = File.create(client, blocks=[Block.CreateRequest(text="Gimme an X!"), Block.CreateRequest(text="Gimme a Y!")])
    tag_task = test_file.tag(trainable_tagger.handle)
    tag_task.wait()

    test_file.refresh()
    assert len(test_file.blocks) == 2
    assert len(test_file.blocks[0].tags) == 2
    print(test_file.blocks[0].tags[0].value.get(TagValue.STRING_VALUE.value))
    assert "X!" in test_file.blocks[0].tags[0].value.get(TagValue.STRING_VALUE.value)
    assert len(test_file.blocks[1].tags) == 2
    print(test_file.blocks[1].tags[0].value.get(TagValue.STRING_VALUE.value))
    assert "Y!" in test_file.blocks[1].tags[0].value.get(TagValue.STRING_VALUE.value)