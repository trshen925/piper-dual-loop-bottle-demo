# Piper bottle-on-coaster Sim2Sim case

Self-contained MuJoCo scene, assets, executable Python skill, and four frozen evaluation cases. The scene has a Piper arm, bottle, coaster, and fixed/wrist cameras. No model checkpoint or private dataset is required.

## Run on another server

```bash
git clone https://github.com/trshen925/piper-dual-loop-bottle-demo.git
cd piper-dual-loop-bottle-demo
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
bash run.sh
```

For the SimEx-style tool/API/task-script path, run:

```bash
MUJOCO_GL=egl python run_cap.py --out outputs/cap_trial_001
```

The package now has two reproducible paths. `run.sh` preserves the original oracle-position diagnostic. `run_cap.py` uses `agent_script.py` → editable `toolbox.py` → fixed `robot_api.py`/`sim_adapter.py`, with camera RGBD and measured robot state as the toolbox inputs. The simulator exposes object state only to its reset/scoring harness.

The run writes `cases.json`, `results.json`, and one metric `*.npz` per case and control into a new `outputs/` directory. To pick an output path: `bash run.sh /path/to/new-output`. Existing directories are rejected. A full run is 16 MuJoCo rollouts and needs no GPU or display.

## What this tests

The four cases are nominal bottle pose, y+16 mm development counterexample, y+18 mm held-out pose, and x+8 mm held-out pose. The four controls are:

| Control | Scene-model bottle XY | Executable skill target XY |
| --- | --- | --- |
| `fixed` | nominal | nominal |
| `model_only` | updated | nominal |
| `skill_only` | nominal | oracle observed pose |
| `adaptive` | updated | updated model pose |

Both `skill_only` and `adaptive` receive exact simulated bottle pose. This case tests the skill update and scene-model bookkeeping, **not camera perception or a real robot deployment**. The outcome on the source server was: all controls passed nominal; fixed/model-only failed y+16 and y+18 mm; skill-only/adaptive passed them; all controls failed x+8 mm because penetration exceeded 1.5 mm. The two updated controls had identical performance, so this case does not yet demonstrate an independent dual-loop gain.

In the new camera-driven path, the program reported visual placement on all four cases. The simulator's physical gate passed nominal and y+16/y+18 mm; it rejected x+8 mm for 3.14 mm penetration. This disagreement is retained in `results.json`. The image detector is calibrated locally to this simulated scene, so this remains a Sim2Sim result.

MuJoCo produces continuous absolute position targets for J1–J6 and joint7/joint8. Policy-interface gripper aperture is `joint7 - joint8` in metres. The action code and scene are simulation only: **do not send `d.ctrl` values directly to Piper hardware**. A robot test needs measured joint state, SDK aperture calibration, TCP and camera calibration, collision limits, and an external stop/approval procedure.

## Files

- `demo.py`: programmatic pick-and-place skill, IK, four controls, metrics.
- `robot_api.py`, `sim_adapter.py`: fixed low-level API and MuJoCo adapter.
- `toolbox.py`, `skill.md`: editable perception/control primitives and measured limits.
- `agent_script.py`, `run_cap.py`: short task program and SimEx-style evaluation harness.
- `real_adapter_template.py`: explicit boundary for a server-specific Piper driver and camera calibration.
- `cases.json`: frozen conditions.
- `scene/scene.xml`: original Real2Gym scene with relative paths.
- `scene/assets/`: mesh and texture files used by the scene.
- `scene/asset_manifest.json`: asset hashes and source names.
- `package_scene.py`: packaging recipe for recreating the portable scene from the original local XML.
- `licenses/piper_description_LICENSE`: upstream Piper mesh license.

The portable scene preserves the original collision geometry and rendering assets. All paths are relative to the repository. The original scene and experiment details are documented in the co_adaption workspace. This package deliberately omits historical run outputs and the 3 GB neural-policy checkpoint.
