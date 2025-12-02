from rlbench.environment import Environment
from rlbench.action_modes.action_mode import JointPositionActionMode
from rlbench.tasks import ReachTarget

action_mode = JointPositionActionMode()

env = Environment(action_mode)
env.launch()

task = env.get_task(ReachTarget)
_, obs = task.reset()

print("Loaded ReachTarget")
print("Image shape:", obs.front_rgb.shape)
