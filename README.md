# Novus-Task2

## Implementation
The required implementation is in encoder_block.py. It is the vanilla transformer encoder implementation that has been used in BERT and ViT as well. Of course it is inferior to the optimized torch implementation such as [flexattention](https://pytorch.org/blog/flexattention/). The torch implementation also covers many edge cases such as different activations, different normalization layers, their orders etc... The one I implemented is pretty standart except the activation which is Gelu that has been popularized by the BERT which uses instead of RELU.

## Simple yet fun test for encoder block.
Since the implementation we have is simple compared to DL frameworks, it requires more effort to adjust them towards. One example is that we use plain Linear layers in MLP block however official torch implementation uses NonDynamicallyQuantizableLinear to avoid quantizing MHA and there are more edge cases similar to this. 

One cool experiment is to learn sorting number with encoder block. We could do this since encoder block is bidirectional that allows us to gather all information to do this operation.In order to this, we have to build the encoder block with encoder layers, so we could have deeper network.Additionally, we need a network to map inputs to tokens and positional encoding layer to distinguish between the order of the tokens. All of these modules are implemented in additional_blocks_for_test.py. Training is done in test_encoder_block.py. By this setup, we can achieve 97& accuracy on arrays of length 16 that includes numbers up to 100.

#### Running Training
```
python -m venv task2env
source task2env/bin/activate
pip install -r requirements.txt
python test_encoder_block.py
```

