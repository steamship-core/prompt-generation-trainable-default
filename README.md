# Trainable Text Generation (prompt-generation-trainable-default)

This project contains a Steamship Tagger that trains an AutoML classifier using Vertex AI and applies it to new blocks. 

## Configuration

This plugin must be configured with the following fields:

| Parameter | Description | DType | Required |
|-------------------|----------------------------------------------------|--------|--|
| max_words | Whether to output single or multiple labels | number | True |
| tag_kind | The `kind` field for tags to train on. | string | True |
| temperature | Controls randomness. Lower values produce higher likelihood / more predictable results; higher values produce more variety. Values between 0-1. | number | False |

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
generator = ship.use_plugin('prompt-generation-trainable-default', config={
  "tag_kind": "generation_training_example",
  "max_words": 100
})

# Add tagged files to Steamship

# Now train the plugin
training_request = TrainingParameterPluginInput(
    plugin_instance=generator.handle,
    export_plugin_input=ExportPluginInput(
        plugin_instance=exporter.handle, type="file", query="blocktag",
    ),
    training_params={},
)

train_result = generator.train(training_request)

# This plugin will take approximately 6 hours to train end-to-end -- this is a result of
# how Google Vertex AI works. When it is complete, the task status will be reported as .succeeded.

tag_task = generator.tag("I like how easy this is!")

```
