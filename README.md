# C3D Gait Event Detection

This project contains a Python script for adding gait events to C3D files from motion capture data.

The main script is [gait_event_detection_c3d_files.py](gait_event_detection_c3d_files.py).

## What the script does

- Reads one or more C3D files.
- Detects Foot Strike and Foot Off from force plate data.
- Uses separate force thresholds:
  - Foot Strike: 10 N
  - Foot Off: 5 N
- Detects the left or right leg from the selected main force plate.
- Adds the following Foot Strike from kinematics using an O'Connor-style approach.
- Writes the new C3D file with added gait events.
- Saves incomplete gait cycles in a separate folder.

## Output folders

The script creates these folders next to the input files:

- `added_gait_events`
- `added_gait_events/incomplete_gait_cycle`

## File naming

The output file keeps the condition name from the input file and adds:

- the detected side, `left` or `right`
- a trial number

Examples:

- `Walking_Pref 3.c3d` -> `Walking_Pref_left_1.c3d`
- `Running_Fixed_right_2.c3d` -> `Running_Fixed_right_1.c3d`

## Requirements

- Packages:
  - `ezc3d`
  - `numpy`
  - `scipy`
  - `tkinter` (included with standard Python on Windows)

## Install Dependencies

If you want to install the required packages with `pip`, run:

```powershell
pip install -r requirements.txt
```

If you prefer Conda, first activate the environment and then install the packages there.

## How to run

Run the script:

```powershell
python gait_event_detection_c3d_files.py
```

## Using the GitHub Repository

If you download this repository from GitHub, keep the script, README, and
`requirements.txt` in the same folder. Then install the dependencies and run
the script from that folder.

## Workflow

1. Select one or more C3D files.
2. Choose the main force plate for naming.
3. The script processes all available force plates.
4. The script writes the new files into `added_gait_events`.

## Notes

- The script is intended for gait trials with force plates and marker data.
- If the analyzed leg does not contain a complete gait cycle, the file is saved in `added_gait_events/incomplete_gait_cycle`.
- The script uses a simple kinematic step based on heel and toe markers and the O'Connor method.

## License

No license has been added yet.
