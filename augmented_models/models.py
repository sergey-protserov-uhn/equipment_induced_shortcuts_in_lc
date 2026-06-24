# Model variants adapted from https://huggingface.co/docs/transformers/en/model_doc/segformer

SEGFORMER_MODEL_VARIANT_CONFIG_OVERRIDES = {
    "sanity_check": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [2, 4, 10, 16],
        "decoder_hidden_size": 16,
    },
    "MiT-b0": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [32, 64, 160, 256],
        "decoder_hidden_size": 256,
    },
    "our_MiT-b0.5": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [48, 96, 240, 384],
        "decoder_hidden_size": 256,
    },
    "MiT-b1": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [64, 128, 320, 512],
        "decoder_hidden_size": 256,
    },
    "our_MiT-b1.5_1": {
        "depths": [3, 4, 6, 3],
        "hidden_sizes": [64, 128, 320, 512],
        "decoder_hidden_size": 256,
    },
    "our_MiT-b1.5_2": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [64, 128, 320, 512],
        "decoder_hidden_size": 512,
    },
    "MiT-b2": {
        "depths": [3, 4, 6, 3],
        "hidden_sizes": [64, 128, 320, 512],
        "decoder_hidden_size": 768,
    },
}
