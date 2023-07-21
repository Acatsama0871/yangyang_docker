from jax.lib import xla_bridge
from rich import print

# jax test
print("[red]Jax GPU test:[/red]")
print(xla_bridge.get_backend().platform)
