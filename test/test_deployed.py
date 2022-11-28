import json
import os
import random
import string

from steamship import Block, File, PluginInstance, Tag
from steamship.client import Steamship
from steamship.plugin.inputs.export_plugin_input import ExportPluginInput
from steamship.plugin.inputs.training_parameter_plugin_input import \
    TrainingParameterPluginInput

PLUGIN_HANDLE = 'trainable-tagger-default'
EXPORTER_HANDLE = "signed-url-exporter"
KEYWORDS = ["product", "coupon"]

def random_name() -> str:
    """Returns a random name suitable for a handle that has low likelihood of colliding with another.

    Output format matches test_[a-z0-9]+, which should be a valid handle.
    """
    letters = string.digits + string.ascii_letters
    return f"test_{''.join(random.choice(letters) for _ in range(10))}".lower()  # noqa: S311


def create_file_example(n: int) -> File.CreateRequest:
    return File.CreateRequest(blocks=[
        Block.CreateRequest(text=f'This is example {n} of birds',
                            tags=[
                                Tag(kind='classification', name='birds')
                            ]
                            ),
        Block.CreateRequest(text=f'This is example {n} of alligators',
                            tags=[
                                Tag.CreateRequest(kind='classification', name='alligators')
                            ]
                            )
    ])

def create_file_examples(n: int) -> [File.CreateRequest]:
    return [create_file_example(i) for i in range(n)]

def test_e2e_trainable_tagger_lambda_training():
    client = Steamship(profile="testLocal")
    exporter_plugin = PluginInstance.create(
        client=client,
        handle=EXPORTER_HANDLE,
        plugin_handle=EXPORTER_HANDLE,
        fetch_if_exists=True,
    )

    folder = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(folder, '..', 'test_data', 'test_config.json'), 'r') as config_file:
        config = json.load(config_file)

    HANDLE = random_name()
    trainable_tagger = PluginInstance.create(
        client=client,
        handle=HANDLE,
        plugin_handle=PLUGIN_HANDLE,
        config=config,
        fetch_if_exists=True,
    )

    for file_request in create_file_examples(20):
        File.create(client, blocks=file_request.blocks)

    # Now train the plugin
    training_request = TrainingParameterPluginInput(
        plugin_instance=HANDLE,
        export_plugin_input=ExportPluginInput(
            plugin_instance=EXPORTER_HANDLE, type="file", query="blocktag",
        ),
        training_params={},
    )

    train_result = trainable_tagger.train(training_request)

    # This will take ~5 hours!

    # train_result.wait()
    # assert train_result.state != TaskState.failed

    # At this point, the PluginInstance will have written a parameter file to disk. We should be able to
    # retrieve it since we know that it is tagged as the `default`.

    # checkpoint = ModelCheckpoint(
    #     client=client,
    #     handle="default",
    #     plugin_instance_id=trainable_tagger.id,
    # )
    # checkpoint_path = checkpoint.download_model_bundle()
    # assert checkpoint_path.exists()
    #
    # return
    #
    # keyword_path = Path(checkpoint_path) / TestTrainableTaggerModel.KEYWORD_LIST_FILE
    # assert keyword_path.exists()
    # with open(keyword_path, "r") as f:
    #     params = json.loads(f.read())
    #     assert params == KEYWORDS
    #
    # logging.info("Waiting 15 seconds for instance to deploy.")
    # import time
    #
    # time.sleep(15)

    # If we're here, we have verified that the plugin instance has correctly recorded its parameters
    # into the pluginData bucket under a path unique to the PluginInstnace/ModelCheckpoint.

    # Now we'll attempt to USE this plugin. This plugin's behavior is to simply tag any file with the
    # tags that parameter it. Since those tags are (see above) ["product", "coupon"] we should expect
    # this tagger to apply those tags to any file provided to it.

    # First we'll create a file
    # test_doc = "Hi there"
    # res = client.tag(doc=test_doc, plugin_instance=tagger_instance.handle)
    # res.wait()
    # assert res.error is None
    # assert res.data is not None
    # assert res.data.file is not None
    # assert res.data.file.tags is not None
    # assert len(res.data.file.tags) == len(KEYWORDS)
    # assert sorted([tag.name for tag in res.data.file.tags]) == sorted(KEYWORDS)
