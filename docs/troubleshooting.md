# Troubleshooting

## I cannot see the `HW` hotspot

- Wait a little longer after boot. The Wi-Fi service can take time to come up.
- Make sure the robot battery is charged enough to boot cleanly.
- If you already configured shared Wi-Fi, the robot may be trying to join that network instead of exposing the hotspot.

## SSH to `192.168.149.1` fails

- Confirm your laptop is actually connected to the `HW` hotspot.
- Try `ping 192.168.149.1` first.
- Reboot the robot and wait for the hotspot to appear again.
- If your image uses different defaults, check the official Hiwonder setup guide linked in the README.

## The `nmcli` command disconnects SSH

That is expected. The robot is leaving hotspot mode and joining another network. Reconnect your laptop to the shared Wi-Fi, find the new robot IP, and SSH back in.

## I do not know the new robot IP

Try one of these:

- router admin page or DHCP lease list
- WonderPi app or other vendor tooling
- local display and keyboard on the robot, then `hostname -I`

If you still cannot find it, reconnect through the hotspot and retry the Wi-Fi steps.

## `python -m client.cli --help` works, but the full client still fails

That usually means one of the laptop runtime dependencies is missing. Reinstall them:

```bash
pip install -r requirements-laptop.txt
```

Also make sure you are running the client locally on the laptop desktop session, not inside SSH.

## The robot server fails with `ros_robot_controller_sdk` missing

Run the server on the TurboPi image that ships with the robot, or on an environment where the Hiwonder SDK is already installed. This dependency is hardware-specific and is not installed from `pip`.

## The robot server fails with `cv2` missing

Most TurboPi images already provide OpenCV. If yours does not, fix the robot image first instead of assuming `pip install opencv-python` will be reliable on the robot.

## The camera does not open

- Make sure another robot-side demo is not already using the same camera device.
- Reboot the robot if the camera was left in a bad state by another app.
- Check the server logs for camera-open errors.

## The robot stops moving by itself

That is usually the motor watchdog doing its job. If the server stops receiving commands for about half a second, it stops the motors for safety.

Common causes:

- Wi-Fi dropped
- the laptop client crashed
- the server is not actually running

## The helper script does not run on Windows

`bash scripts/deploy_server.sh start` and `bash scripts/deploy_server.sh deps` need a Bash shell. Use Git Bash or WSL, or SSH into the robot and run the commands directly there.

## Accepted episodes save badly or look inconsistent

Check these first:

- make sure the robot was actually receiving commands
- make sure battery voltage is healthy
- make sure the server was not dropping camera frames

The recorder skips frames when motor commands fail instead of silently saving mismatched labels, but unstable Wi-Fi can still reduce collection quality.

## The Intent-CNN trainer says it is using only one session

That warning means training can still run, but true validation is skipped because there is only one recording session available.

Fix:

- collect at least one more `session_YYYYMMDD_HHMMSS` under `data/turbopi_intent_cnn/episodes/`
- then train from the full Intent-CNN episodes root instead of one specific session folder

## `intent_cnn_policy.drive` says the task is unknown

That checkpoint only knows the task strings that were present in its training data.

Fixes:

- pass a task string that really appears in the dataset used for that checkpoint
- inspect the checkpoint metadata or training summary to confirm the saved task vocabulary
- retrain if you need a newly added custom task to be supported

## LeRobot export fails

Check these common causes:

- `pip install -r requirements-export.txt` was never run
- `ffmpeg` is not installed or not on your `PATH`
- an episode folder is missing either `video.mp4` or `data.parquet`
- the MP4 frame count does not match the Parquet row count

Useful command:

```bash
python scripts/export_lerobot.py --help
```

If you are exporting older recordings, keep the default:

```bash
--state-source shifted_action
```

That rebuilds `observation.state` from the previous action and is the safest option for training.

If export succeeds but the console looks noisy:

- `libx264` lines are normal video-encoder logs, not failures
- `torchcodec is not available ... falling back to pyav` is a fallback warning, not a failed export
- LeRobot may store exported frames in one chunked video file, so the visible video length is based on accepted frames at the dataset FPS

## `smolvla_policy.drive` says the checkpoint is incomplete or corrupted

That usually means the copied `pretrained_model/model.safetensors` file was truncated during download or transfer.

Fixes:

- re-copy the full `pretrained_model/` folder from the training machine
- avoid partial browser downloads for multi-GB model files
- verify the local file size before trying again

## `smolvla_policy.drive` loads but the robot barely moves

Common causes:

- `vx-cap` or `omega-cap` are too low
- smoothing is too high for the robot to react quickly
- the task prompt is valid, but the model still learned a weak command scale from the dataset

Typical tuning knobs:

- increase `--vx-cap`
- increase `--omega-cap`
- reduce `--smoothing`
- keep `--min-vy 0` unless you really want forced sideways motion

## `torchcodec is not available ... falling back to pyav`

That warning is noisy but usually harmless.

It means LeRobot or the local decoder is using PyAV instead of TorchCodec on your platform.

## The robot returns to hotspot mode after reboot

That usually means the shared Wi-Fi settings were temporary or the persistent configuration is not correct yet.

Fix:

1. Reconnect to the hotspot.
2. SSH to `192.168.149.1`.
3. Re-run the `nmcli` connection.
4. Update `~/hiwonder-toolbox/wifi_conf.py` so shared Wi-Fi is the saved default.

## Windows says the full client or drive loop behaves strangely inside SSH

Run the laptop-side tools from the local desktop session, not from a remote SSH shell.

That especially matters for:

- keyboard teleop
- live drive loops
- local camera/inference debugging

## Battery and power notes

- Low battery causes strange behavior before it causes a full shutdown.
- Charge the pack if motion becomes inconsistent or the robot reboots unexpectedly.
- Start longer recording sessions only after checking battery health.
