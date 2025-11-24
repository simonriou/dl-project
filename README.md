# dl-project

## Models overview
- `model_1-3`: Trained on babble_16k.wav only, mainly to test the architecture (implementation concerns mostly)
- `model_4`: Trained on multiple babble noises.
- `model_5`: Trained on multiple babble noises, added skip connections.
- `model_6`: Same for training, but using a soft mask (IRM) instead of binary mask (IBM).
- `model_7`: Same as model_6, but tuned the initial learing rate a bit.
- Future models: phase-aware masks?