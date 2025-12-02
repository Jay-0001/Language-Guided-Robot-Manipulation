from rlbench.environment import Environment
from rlbench.action_modes.action_mode import JointPositionActionMode
from rlbench.tasks import ReachTarget

env = Environment(JointPositionActionMode(), headless=False)
env.launch(launch_coppeliasim=False)  # ← THIS IS THE FIX

task = env.get_task(ReachTarget)
descs, obs = task.reset()

print("Loaded ReachTarget")
print("Image shape:", obs.front_rgb.shape)

env.shutdown()
