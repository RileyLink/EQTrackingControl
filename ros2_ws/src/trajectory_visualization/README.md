# Playing exported policy rollouts in RViz

First generate the rollout files from the repository root with `python -m rl.eval_vis`. The exporter supports full, original strong, corrected strong, and weak policies for both quadrotor and Astrobee; see [checkpoint selection and export commands](../../../rl/README.md#ros-rollout-export-eval_vispy).

The current player has two policy slots plus the reference. Select the displayed modes when exporting, for example `--rviz-pair corrected_strong weak`. The first mode occupies the existing `baseline` slot; the second occupies the `equivariant`/`symmetry` slot. These slot names do not change the checkpoint's observation mode. Every requested mode is also saved separately.

For example, after training checkpoints in the default folders, run from the repository root:

```bash
python -m rl.eval_vis --env quadrotor --reference random_walk --rviz-pair corrected_strong weak
```

The data are written to `ros2_ws/src/trajectory_visualization/trajectory_visualization/data/quadrotor/`. For Astrobee, use `--env astrobee`, which writes its own `data/astrobee/` folder. Quaternions use `x, y, z, w` ordering. The readers use the exported trajectory length rather than assuming 2,000 samples.

In a terminal with ROS2 installed and its environment sourced:

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
rviz2 -d src/trajectory_visualization/quadrotor.rviz
```

If Colcon is missing on Ubuntu, install it with `sudo apt install python3-colcon-common-extensions`. For Astrobee, the supplied RViz files are `astrobee_baseline.rviz` and `astrobee_symmetry.rviz`; there is no `astrobee.rviz`. Choose one and enable the desired reference/baseline/symmetry displays.

In another ROS2 terminal, also starting in `ros2_ws` and sourcing its setup:

```bash
source install/setup.bash
ros2 run trajectory_visualization eval_visualize --args quadrotor
```

Replace `quadrotor` with `astrobee` for Astrobee. The player waits for Enter before playback and loops when it reaches the exported length. It publishes reference/baseline/symmetry transforms and paths. Its existing defaults play 20 samples per simulated second at half real-time speed. Check `manifest.json` to identify which mode occupies each slot and `metrics.csv` for tracking errors.

The player reads relative to the ROS workspace's source folder, so run it from `ros2_ws`. If you used a custom exporter `--output-dir`, copy that export's pair/reference files into the appropriate `data/<robot>/` folder first. Stop and restart the player after regenerating files; it loads them at startup.

For the original particle visualization, export with `--env particle` and run `ros2 run trajectory_visualization eval_visualize_particle --args particle`, using one of the supplied particle RViz files.
