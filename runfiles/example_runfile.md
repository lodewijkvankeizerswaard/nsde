# Example runfile that trains a normalizing flow on the 2D simulators
# using affine autoregressive normalizing flows.

## Affine autoregressive
# Direct
python main.py --problem 2d --loss-function marginal --model naive --marginal affine-autoregressive --device cuda --problem2d-marginal mixture --marginal-layers 12
python main.py --problem 2d --loss-function marginal  --model naive --marginal affine-autoregressive --device cuda --problem2d-marginal torus --marginal-layers 12
