from steamship import File, Block, Tag
from steamship.data import TagValue
import json

def main():
    blocks = []
    for letter in "STEAMSHIP":
        block = Block(text=f'Gimme a {letter}!')
        block.tags = [Tag(kind="training_generation", startIdx=0, endIdx=len(block.text), value={TagValue.STRING_VALUE.value: letter })]
        blocks.append(block)

    file = File(blocks=blocks)
    with open("../test_data/test_training_data.jsonl", 'w') as output:
        output.write(json.dumps(file.dict()))

if __name__ == "__main__":
    main()