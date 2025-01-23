import jax
from jax import export
import jax.numpy as jnp

# Load StableHLO text
with open("nanogpt.stablehlo", "rb") as f:
    stablehlo = f.read()

# Rebuild the exported function
rehydrated = export.deserialize(stablehlo)
def callee(y):
    return 3 * rehydrated.call(y * 4)

# Execute with JAX inputs
x = jnp.zeros((1,128), dtype=jnp.int32) # match original shape
result = callee(x)
print(result)
