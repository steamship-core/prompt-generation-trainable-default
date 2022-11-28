# Vertex AI Classifier (trainable-tagger-default)

This project contains a Steamship Tagger that trains an AutoML classifier using Vertex AI and applies it to new blocks. 

## Configuration

This plugin must be configured with the following fields:

| Parameter | Description | DType | Required |
|-------------------|----------------------------------------------------|--------|--|
| single_or_multi_label | Whether to output single or multiple labels | string | True |
| tag_kind | The `kind` field for tags to output. Tag name will be the output label. | boolean | True |
| include_tag_names | Whether to train on a subset of tag names (csv string). If empty, all tags of the provided kind will be used for training | boolean | False |

## Getting Started

### Usage

To authenticate with Steamship, install the Steamship CLI with:

```bash
> npm install -g @steamship/cli
```

And then login with:

```bash
> ship login
```

Finally, use the Tagger:

```python
from steamship import Steamship
from steamship.plugin.inputs.training_parameter_plugin_input import TrainingParameterPluginInput
from steamship.plugin.inputs.export_plugin_input import ExportPluginInput

ship = Steamship()  # Without arguments, credentials in ~/.steamship.json will be used.

exporter = ship.use_plugin('signed-url-exporter')
tagger = ship.use_plugin('trainable-tagger-default', config={
  "single_or_multi_label": "single",
  "tag_kind": "classification"
})

# Add tagged files to Steamship

# Now train the plugin
training_request = TrainingParameterPluginInput(
    plugin_instance=tagger.handle,
    export_plugin_input=ExportPluginInput(
        plugin_instance=exporter.handle, type="file", query="blocktag",
    ),
    training_params={},
)

train_result = tagger.train(training_request)

# This plugin will take approximately 6 hours to train end-to-end -- this is a result of
# how Google Vertex AI works. When it is complete, the task status will be reported as .succeeded.

tag_task = tagger.tag("I like how easy this is!")

```
