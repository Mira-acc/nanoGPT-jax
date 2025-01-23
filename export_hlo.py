import jax
from jax import export
import jax.numpy as jnp

from jax.lib import xla_client

from model import GPTConfig, GPT

def main():
    conf = GPTConfig()
    model = GPT(conf)
    x = jnp.zeros((1, 128), dtype=jnp.int32)  # or whatever shape
    key = jax.random.PRNGKey(0)  # Create proper random key
    params = model.init(key, x, train=False)

    def model2(x):
        return model.apply(params, x, train=False)
    jitted = jax.jit(model2)
    compiled = jitted.lower(x)
    hlo_text = compiled.compiler_ir(dialect="hlo").as_hlo_text()

    # Save to file
    with open("nanogpt.hlo", "w") as f:
        f.write(hlo_text)

    # https://jax.readthedocs.io/en/latest/export/export.html
    exp = export.export(jitted)(x)
    stablehlo_text = exp.mlir_module()
    stablehlo_bytecode = exp.serialize()
    with open("nanogpt.stablehlo.txt", "w") as f:
        f.write(stablehlo_text)
    with open("nanogpt.stablehlo", "wb") as f:
        f.write(stablehlo_bytecode)

main()
