import glob
import os
import re 
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
from scipy.signal import butter, filtfilt

"""Gait event detection for C3D files.

Overview
--------
This script reads one or more C3D files and adds gait events.

1. Foot Strike and Foot Off are found from force plate data using
	separate thresholds (Strike=10 N, Off=5 N).
2. For each detected force-plate contact, the script decides whether
	the contact is from the left or right leg.
3. For each stance, the script adds events from force Foot Strike to
	force Foot Off, then adds the following Foot Strike from kinematics
	using the midpoint between heel vertical-velocity local minima and
	toe vertical-velocity local minima after toe-off. Similar to the method from O'Connor et al. 2007 
	(https://www.sciencedirect.com/science/article/pii/S0966636206001068).
4. Events are written back to a new C3D file.
5. The user chooses one main force plate for naming.
6. Output files are named with left/right suffix and trial number.
7. New files are saved in a subfolder called added_gait_events.

"""



try:
	import ezc3d  # type: ignore
except ImportError as exc:
	raise ImportError(
		"ezc3d is required. Use the c3d_parser_env interpreter or run: "
		"conda run -n c3d_parser_env python -m pip install ezc3d"
	) from exc


THRESHOLD_STRIKE_N = 10.0
THRESHOLD_OFF_N = 5.0
MIN_SWING_TIME_S = 0.30
MAX_SWING_TIME_S = 2.00


def select_c3d_files(root):
	"""Open a dialog and let the user select one or more C3D files.

	Parameters
	----------
	root : tkinter.Tk
		Hidden Tk root. Included for consistent function interface.

	Returns
	-------
	list[str]
		Selected file paths. Empty list means user cancelled.
	"""
	files = filedialog.askopenfilenames(
		title="Select one or more C3D files",
		filetypes=[("C3D files", "*.c3d"), ("All files", "*.*")],
	)
	return list(files)


def get_plate_count(c3d_obj):
	"""Read the number of force plates from the C3D file.

	Parameters
	----------
	c3d_obj : dict-like
		C3D object loaded by ezc3d.

	Returns
	-------
	int
		Number of plates from FORCE_PLATFORM:USED.
	"""
	fp = c3d_obj["parameters"].get("FORCE_PLATFORM", {})
	used = fp.get("USED", {}).get("value", [0])
	used_arr = np.asarray(used).astype(int).reshape(-1)
	return int(used_arr[0]) if used_arr.size else 0


def ask_plate_index(root, min_plate, max_plate):
	"""Ask the user to choose the main force plate for naming.

	Parameters
	----------
	root : tkinter.Tk
		Hidden Tk root.
	min_plate : int
		Smallest allowed plate number.
	max_plate : int
		Largest allowed plate number.

	Returns
	-------
	int | None
		Selected plate number (1-based), or None if cancelled.
	"""
	while True:
		idx = simpledialog.askinteger(
			"Main Force Plate",
			f"Select main force plate for file naming ({min_plate}-{max_plate}):",
			minvalue=min_plate,
			maxvalue=max_plate,
			parent=root,
		)
		if idx is None:
			return None
		if min_plate <= idx <= max_plate:
			return idx


def get_force_plate_channels(c3d_obj, plate_number_1based):
	"""Return analog channel indices for one force plate.

	The C3D parameter FORCE_PLATFORM:CHANNEL is read and converted
	from 1-based channel numbering (C3D convention) to 0-based
	channel numbering (NumPy/Python convention).

	Parameters
	----------
	c3d_obj : dict-like
		C3D object loaded by ezc3d.
	plate_number_1based : int
		Force plate number, counted from 1.

	Returns
	-------
	np.ndarray
		Zero-based analog channel indices for the selected plate.
	"""
	fp = c3d_obj["parameters"].get("FORCE_PLATFORM", {})
	if "CHANNEL" not in fp:
		raise ValueError("FORCE_PLATFORM:CHANNEL not found in C3D parameters.")

	ch = np.asarray(fp["CHANNEL"]["value"], dtype=int)
	if ch.ndim != 2:
		raise ValueError("FORCE_PLATFORM:CHANNEL has unexpected dimensions.")

	nplates = get_plate_count(c3d_obj)
	plate_idx = plate_number_1based - 1

	if ch.shape[0] == nplates and ch.shape[1] >= 3:
		channels = ch[plate_idx, :]
	elif ch.shape[1] == nplates and ch.shape[0] >= 3:
		channels = ch[:, plate_idx]
	else:
		raise ValueError(
			f"Cannot map CHANNEL shape {ch.shape} to {nplates} force plates."
		)

	# C3D channel numbering is 1-based.
	channels_zero_based = np.asarray(channels, dtype=int) - 1
	if np.any(channels_zero_based < 0):
		raise ValueError("Invalid channel mapping in FORCE_PLATFORM:CHANNEL.")

	return channels_zero_based


def get_force_plate_center(c3d_obj, plate_number_1based):
	"""Compute geometric center of one force plate.

	The center is the mean of the four plate corner coordinates
	stored in FORCE_PLATFORM:CORNERS.

	Parameters
	----------
	c3d_obj : dict-like
		C3D object loaded by ezc3d.
	plate_number_1based : int
		Force plate number, counted from 1.

	Returns
	-------
	np.ndarray
		Plate center as [x, y, z] in lab coordinates.
	"""
	fp = c3d_obj["parameters"].get("FORCE_PLATFORM", {})
	if "CORNERS" not in fp:
		raise ValueError("FORCE_PLATFORM:CORNERS not found in C3D parameters.")

	corners = np.asarray(fp["CORNERS"]["value"], dtype=float)
	nplates = get_plate_count(c3d_obj)
	plate_idx = plate_number_1based - 1

	if corners.ndim != 3:
		raise ValueError("FORCE_PLATFORM:CORNERS has unexpected dimensions.")

	# Expected shape is usually (3, 4, nplates).
	if corners.shape[0] == 3 and corners.shape[1] == 4 and corners.shape[2] == nplates:
		plate_corners = corners[:, :, plate_idx]
	elif corners.shape[0] == nplates and corners.shape[1] == 3 and corners.shape[2] == 4:
		plate_corners = corners[plate_idx, :, :]
	else:
		raise ValueError(
			f"Cannot map CORNERS shape {corners.shape} to {nplates} force plates."
		)

	return np.mean(plate_corners, axis=1)


def detect_force_events(fz, strike_threshold_n, off_threshold_n):
	"""Detect Foot Strike and Foot Off from vertical force signal.

	A hysteresis rule is used:
	- Foot Strike when abs(Fz) rises to or above strike threshold.
	- Foot Off when abs(Fz) falls to or below off threshold.
	This avoids unstable switching near one single threshold.

	Parameters
	----------
	fz : np.ndarray
		Vertical force time series.
	strike_threshold_n : float
		Threshold for Foot Strike in Newton.
	off_threshold_n : float
		Threshold for Foot Off in Newton.

	Returns
	-------
	tuple[list[int], list[int]]
		(strike_samples, off_samples), both in analog sample index.
	"""
	if off_threshold_n > strike_threshold_n:
		raise ValueError("off_threshold_n must be less than or equal to strike_threshold_n.")

	abs_fz = np.abs(np.asarray(fz, dtype=float))
	if abs_fz.size == 0:
		return [], []

	strike_samples = []
	off_samples = []
	in_contact = bool(abs_fz[0] >= strike_threshold_n)

	if in_contact:
		strike_samples.insert(0, 0)

	for i in range(1, abs_fz.size):
		if (not in_contact) and abs_fz[i] >= strike_threshold_n:
			strike_samples.append(i)
			in_contact = True
		elif in_contact and abs_fz[i] <= off_threshold_n:
			off_samples.append(i)
			in_contact = False

	if in_contact:
		off_samples.append(abs_fz.size - 1)

	return strike_samples, off_samples


def find_point_label_index(c3d_obj, label_name):
	"""Find marker index from POINT:LABELS using name match.

	The match is case-insensitive and ignores outer spaces.

	Parameters
	----------
	c3d_obj : dict-like
		C3D object loaded by ezc3d.
	label_name : str
		Marker label to search for.

	Returns
	-------
	int | None
		Marker index if found, otherwise None.
	"""
	labels = c3d_obj["parameters"]["POINT"]["LABELS"]["value"]
	labels_clean = [str(x).strip().lower() for x in labels]
	target = label_name.strip().lower()
	if target not in labels_clean:
		return None
	return labels_clean.index(target)


def resolve_leg_marker_labels(c3d_obj, leg_context):
	"""Resolve heel and toe marker labels for one leg.

	The function tries common label variants and returns the
	first valid pair for the selected leg.

	Parameters
	----------
	c3d_obj : dict-like
		C3D object loaded by ezc3d.
	leg_context : str
		"Left" or "Right".

	Returns
	-------
	tuple[str, str]
		(heel_label, toe_label)
	"""
	if leg_context == "Right":
		heel_candidates = ["calc_back_right", "heel_right", "rheel"]
		toe_candidates = ["toe_right", "calc_front_right", "rtoe"]
	else:
		heel_candidates = ["calc_back_left", "heel_left", "lheel"]
		toe_candidates = ["toe_left", "calc_front_left", "ltoe"]

	def _first_existing(candidates):
		for name in candidates:
			if find_point_label_index(c3d_obj, name) is not None:
				return name
		return None

	heel_label = _first_existing(heel_candidates)
	toe_label = _first_existing(toe_candidates)

	if heel_label is None or toe_label is None:
		raise ValueError(
			"Could not find required heel/toe markers for selected leg. "
			f"Heel candidates={heel_candidates}, toe candidates={toe_candidates}."
		)

	return heel_label, toe_label


def detect_leg_context_from_plate(
	c3d_obj,
	plate_number,
	strike_sample,
	analog_rate,
	start_time,
	point_rate,
):
	"""Classify one force-plate strike as left or right leg.

	Method
	------
	At the strike time, distance from each heel marker to the selected
	force plate center is compared. The shorter distance wins.

	Parameters
	----------
	c3d_obj : dict-like
		C3D object loaded by ezc3d.
	plate_number : int
		Force plate number, counted from 1.
	strike_sample : int
		Strike sample index in analog data.
	analog_rate : float
		Analog sampling frequency (Hz).
	start_time : float
		Time origin used in the script.
	point_rate : float
		Marker sampling frequency (Hz).

	Returns
	-------
	str
		"Left" or "Right".
	"""
	left_heel_label, _ = resolve_leg_marker_labels(c3d_obj, "Left")
	right_heel_label, _ = resolve_leg_marker_labels(c3d_obj, "Right")
	left_idx = find_point_label_index(c3d_obj, left_heel_label)
	right_idx = find_point_label_index(c3d_obj, right_heel_label)
	if left_idx is None or right_idx is None:
		raise ValueError("Could not find both left and right heel markers.")

	points = c3d_obj["data"]["points"]
	plate_center = get_force_plate_center(c3d_obj, plate_number)
	time_s = start_time + strike_sample / analog_rate
	point_frame = int(round((time_s - start_time) * point_rate))
	point_frame = max(0, min(point_frame, points.shape[2] - 1))

	left_xyz = points[:3, left_idx, point_frame].astype(float)
	right_xyz = points[:3, right_idx, point_frame].astype(float)

	left_dist = None
	right_dist = None
	if np.all(np.isfinite(left_xyz)):
		left_dist = float(np.linalg.norm(left_xyz - plate_center))
	if np.all(np.isfinite(right_xyz)):
		right_dist = float(np.linalg.norm(right_xyz - plate_center))

	if left_dist is None and right_dist is None:
		raise ValueError("Heel markers are invalid at strike times; cannot detect leg side.")
	if left_dist is None:
		return "Right"
	if right_dist is None:
		return "Left"

	return "Left" if left_dist < right_dist else "Right"


def pair_strikes_with_offs(strike_samples, off_samples):
	"""Create stance pairs from strike and toe-off sample lists.

	Each strike is paired with the first toe-off that comes after it.
	If no later toe-off exists, that strike is ignored.

	Parameters
	----------
	strike_samples : list[int]
		Force-strike sample indices.
	off_samples : list[int]
		Force toe-off sample indices.

	Returns
	-------
	list[tuple[int, int]]
		List of (strike_sample, off_sample) pairs.
	"""
	pairs = []
	off_idx = 0
	for strike in strike_samples:
		while off_idx < len(off_samples) and off_samples[off_idx] <= strike:
			off_idx += 1
		if off_idx >= len(off_samples):
			break
		pairs.append((strike, off_samples[off_idx]))
		off_idx += 1
	return pairs


def deduplicate_events(events, tol_s=0.005):
	"""Remove repeated events that are almost at the same time.

	Two events are treated as duplicates when they have the same
	label, same context, and time difference smaller than tolerance.

	Parameters
	----------
	events : list[tuple[float, str, str]]
		(time, label, context)
	tol_s : float, default=0.005
		Tolerance in seconds.

	Returns
	-------
	list[tuple[float, str, str]]
		Cleaned event list sorted by time.
	"""
	seen = set()
	unique = []
	for event_time, label, context in sorted(events, key=lambda x: x[0]):
		key = (round(float(event_time) / tol_s), label, context)
		if key in seen:
			continue
		seen.add(key)
		unique.append((float(event_time), label, context))
	return unique


def interpolate_nans(signal):
	"""Fill missing samples in a 1D signal by linear interpolation.

	Parameters
	----------
	signal : array-like
		Input signal with possible NaN values.

	Returns
	-------
	np.ndarray | None
		Interpolated signal, or None if all values are invalid.
	"""
	signal = np.asarray(signal, dtype=float)
	valid = np.isfinite(signal)
	if not np.any(valid):
		return None
	if np.all(valid):
		return signal.copy()
	idx = np.arange(signal.size)
	out = signal.copy()
	out[~valid] = np.interp(idx[~valid], idx[valid], signal[valid])
	return out


def lowpass_butterworth(signal, fs, cutoff_hz=7.0, order=2):
	"""Apply a low-pass Butterworth filter to one signal.

	Parameters
	----------
	signal : np.ndarray
		Input time series.
	fs : float
		Sampling frequency (Hz).
	cutoff_hz : float, default=7.0
		Cutoff frequency (Hz).
	order : int, default=2
		Filter order.

	Returns
	-------
	np.ndarray
		Filtered signal.
	"""
	nyquist = 0.5 * fs
	norm = cutoff_hz / nyquist
	norm = min(max(norm, 1e-6), 0.999999)
	b, a = butter(order, norm, btype="low")
	return filtfilt(b, a, signal)


def detect_following_foot_strike_oconnor(
	c3d_obj,
	heel_label,
	toe_label,
	toe_off_time,
	start_time,
	point_rate,
):
	"""Detect the next Foot Strike from kinematics after toe-off.

	Method
	------
	1. Use heel and toe vertical marker signals.
	2. Low-pass filter marker trajectories (7 Hz, order 2).
	3. Build vertical velocities.
	4. Search after Toe Off for heel and toe vertical-velocity local minima.
	5. Use the midpoint between the paired minima as Foot Strike.

	Parameters
	----------
	c3d_obj : dict-like
		C3D object loaded by ezc3d.
	heel_label : str
		Heel marker label.
	toe_label : str
		Toe marker label.
	toe_off_time : float
		Toe-off time in seconds.
	start_time : float
		Time origin used in the script.
	point_rate : float
		Marker sampling frequency (Hz).

	Returns
	-------
	float | None
		Detected Foot Strike time in seconds, or None.

	Notes
	-----
	The function name is kept for backward compatibility with the rest of the
	script, but the current implementation no longer applies the original
	O'Connor-style rule. It now uses midpoint timing from heel and toe
	vertical-velocity local minima within the post-toe-off search window.
	"""
	max_pair_gap_s = 0.20
	heel_idx = find_point_label_index(c3d_obj, heel_label)
	toe_idx = find_point_label_index(c3d_obj, toe_label)
	if heel_idx is None or toe_idx is None:
		return None

	points = c3d_obj["data"]["points"]  # 4 x markers x frames
	heel_z = points[2, heel_idx, :].astype(float)
	toe_z = points[2, toe_idx, :].astype(float)

	heel_z = interpolate_nans(heel_z)
	toe_z = interpolate_nans(toe_z)
	if heel_z is None or toe_z is None:
		return None

	try:
		heel_z_f = lowpass_butterworth(heel_z, fs=point_rate, cutoff_hz=7.0, order=2)
		toe_z_f = lowpass_butterworth(toe_z, fs=point_rate, cutoff_hz=7.0, order=2)
	except ValueError:
		# Fallback when data length is too short for filtfilt padding.
		heel_z_f = heel_z
		toe_z_f = toe_z

	dt = 1.0 / point_rate
	heel_vz = np.gradient(heel_z_f, dt)
	toe_vz = np.gradient(toe_z_f, dt)

	toe_off_frame = int(round((toe_off_time - start_time) * point_rate))
	min_sep = int(round(MIN_SWING_TIME_S * point_rate))
	max_sep = int(round(MAX_SWING_TIME_S * point_rate))
	max_pair_gap = max(1, int(round(max_pair_gap_s * point_rate)))

	start_idx = max(toe_off_frame + min_sep, 2)
	end_idx = min(toe_off_frame + max_sep, len(heel_z_f) - 3)
	if start_idx >= end_idx:
		return None

	heel_min_idx = []
	toe_min_idx = []
	for i in range(start_idx, end_idx + 1):
		if heel_vz[i - 1] > heel_vz[i] <= heel_vz[i + 1]:
			heel_min_idx.append(i)
		if toe_vz[i - 1] > toe_vz[i] <= toe_vz[i + 1]:
			toe_min_idx.append(i)

	if heel_min_idx and toe_min_idx:
		toe_arr = np.asarray(toe_min_idx, dtype=int)
		candidate_midpoints = []
		for h_idx in heel_min_idx:
			d = np.abs(toe_arr - h_idx)
			best = int(np.argmin(d))
			t_idx = int(toe_arr[best])
			if abs(t_idx - h_idx) <= max_pair_gap:
				candidate_midpoints.append(int(round(0.5 * (h_idx + t_idx))))
		if candidate_midpoints:
			return start_time + min(candidate_midpoints) / point_rate

	# Fallback: earliest available local minimum if no midpoint pair is found.
	if heel_min_idx:
		return start_time + heel_min_idx[0] / point_rate
	if toe_min_idx:
		return start_time + toe_min_idx[0] / point_rate

	return None


def load_existing_events(c3d_obj):
	"""Read existing EVENT entries from a C3D object.

	Parameters
	----------
	c3d_obj : dict-like
		C3D object loaded by ezc3d.

	Returns
	-------
	list[tuple[float, str, str]]
		List of (time, label, context) tuples.
	"""
	params = c3d_obj["parameters"]
	if "EVENT" not in params or "USED" not in params["EVENT"]:
		return []

	ev = params["EVENT"]
	used = int(np.asarray(ev["USED"]["value"]).reshape(-1)[0])

	labels = [str(x).strip() for x in ev.get("LABELS", {}).get("value", [])]
	contexts = [str(x).strip() for x in ev.get("CONTEXTS", {}).get("value", [])]
	times = np.asarray(ev.get("TIMES", {}).get("value", np.zeros((2, 0))), dtype=float)

	if times.ndim != 2 or times.shape[0] != 2:
		return []

	used = min(used, len(labels), len(contexts), times.shape[1])
	events = []
	for i in range(used):
		label = labels[i] if i < len(labels) else ""
		context = contexts[i] if i < len(contexts) else ""
		event_time = float(times[1, i])
		events.append((event_time, label, context))

	return events


def write_events(c3d_obj, events):
	"""Append detected events to the C3D EVENT group.

	Events are sorted by time before writing.

	This function uses ezc3d.add_event(), which safely handles
	metadata creation when the file has no EVENT group yet.

	Parameters
	----------
	c3d_obj : dict-like
		C3D object loaded by ezc3d.
	events : list[tuple[float, str, str]]
		(time, label, context)
	"""
	events_sorted = sorted(events, key=lambda x: x[0])

	for event_time, label, context in events_sorted:
		icon_id = 1 if label == "Foot Strike" else 2
		c3d_obj.add_event(
			time=(0, float(event_time)),
			context=str(context),
			label=str(label),
			description="",
			subject="",
			icon_id=icon_id,
			generic_flag=0,
		)


def has_complete_gait_cycle(events, leg_context):
	"""Check if one full cycle exists for the analyzed leg.

	A complete cycle is defined as:
	1. Foot Strike
	2. Foot Off
	3. Following Foot Strike

	Parameters
	----------
	events : list[tuple[float, str, str]]
		(time, label, context)
	leg_context : str
		"Left" or "Right".

	Returns
	-------
	bool
		True if a full Foot Strike -> Foot Off -> Foot Strike sequence exists.
	"""
	leg_events = [(t, l) for t, l, c in sorted(events, key=lambda x: x[0]) if c == leg_context]  # noqa: E741
	if len(leg_events) < 3:
		return False

	for i in range(len(leg_events) - 2):
		if (
			leg_events[i][1] == "Foot Strike"
			and leg_events[i + 1][1] == "Foot Off"
			and leg_events[i + 2][1] == "Foot Strike"
		):
			return True

	return False


def make_output_path(input_path, leg_context, incomplete_cycle=False):
	"""Create output path in add_gait_events with side and trial number.

	Parameters
	----------
	input_path : str
		Original C3D file path.
	leg_context : str
		"Left" or "Right" used in output suffix.
	incomplete_cycle : bool, default=False
		If True, save file in added_gait_events/incomplete_gait_cycle.

	Returns
	-------
	str
		Output file path, for example Walking_FIX_left_1.c3d.
	"""
	source_folder = os.path.dirname(input_path)
	folder = os.path.join(source_folder, "added_gait_events")
	if incomplete_cycle:
		folder = os.path.join(folder, "incomplete_gait_cycle")
	os.makedirs(folder, exist_ok=True)
	stem = os.path.splitext(os.path.basename(input_path))[0]

	# Keep the condition name and remove only trailing side/trial suffixes.
	# Examples:
	# - Walking_Pref 3 -> Walking_Pref
	# - Running_Fixed_left_2 -> Running_Fixed
	normalized_stem = re.sub(r"\s+", "_", stem).strip("_-")
	cleaned_stem = re.sub(r"(?:_(?:left|right))?[_-]?\d+$", "", normalized_stem, flags=re.IGNORECASE)
	cleaned_stem = re.sub(r"_(?:left|right)$", "", cleaned_stem, flags=re.IGNORECASE)
	cleaned_stem = cleaned_stem.strip("_-")
	if not cleaned_stem:
		cleaned_stem = normalized_stem if normalized_stem else stem

	leg_suffix = leg_context.lower()
	pattern = os.path.join(folder, f"{cleaned_stem}_{leg_suffix}_*.c3d")
	existing = glob.glob(pattern)

	max_trial = 0
	rx = re.compile(rf"^{re.escape(cleaned_stem)}_{leg_suffix}_(\d+)\.c3d$", re.IGNORECASE)
	for p in existing:
		m = rx.match(os.path.basename(p))
		if m:
			max_trial = max(max_trial, int(m.group(1)))

	next_trial = max_trial + 1
	return os.path.join(folder, f"{cleaned_stem}_{leg_suffix}_{next_trial}.c3d")


def process_file(
	file_path,
	main_plate_number,
	strike_threshold_n=THRESHOLD_STRIKE_N,
	off_threshold_n=THRESHOLD_OFF_N,
):
	"""Process one C3D file and write a new file with added events.

	Workflow
	--------
	1. Read all force plates.
	2. Detect strike/off from force.
	3. Identify leg side per strike.
	4. Add following kinematic strike.
	5. Save output file in add_gait_events.

	Parameters
	----------
	file_path : str
		Input C3D file.
	main_plate_number : int
		Main force plate used to determine left/right file suffix.
	strike_threshold_n : float, default=THRESHOLD_STRIKE_N
		Threshold for Foot Strike in Newton.
	off_threshold_n : float, default=THRESHOLD_OFF_N
		Threshold for Foot Off in Newton.

	Returns
	-------
	dict
		Summary dictionary with counts and output path.
	"""
	c3d_obj = ezc3d.c3d(file_path)

	point_rate = float(c3d_obj["header"]["points"]["frame_rate"])
	analog_rate = float(c3d_obj["header"]["analogs"]["frame_rate"])
	first_frame = int(c3d_obj["header"]["points"]["first_frame"])
	start_time = first_frame / point_rate
	plate_count = get_plate_count(c3d_obj)
	if plate_count < 1:
		raise ValueError("No force plates found in file.")

	analog = c3d_obj["data"]["analogs"][0, :, :]  # channels x samples
	detected_events = []
	n_force_strikes = 0
	n_force_offs = 0
	main_plate_leg = None

	for plate_number in range(1, plate_count + 1):
		channels = get_force_plate_channels(c3d_obj, plate_number)
		if channels.size < 3:
			continue

		fz = analog[int(channels[2]), :]
		strike_samples, off_samples = detect_force_events(
			fz=fz,
			strike_threshold_n=strike_threshold_n,
			off_threshold_n=off_threshold_n,
		)
		pairs = pair_strikes_with_offs(strike_samples, off_samples)

		if plate_number == main_plate_number and pairs and main_plate_leg is None:
			main_strike_sample = pairs[0][0]
			main_plate_leg = detect_leg_context_from_plate(
				c3d_obj=c3d_obj,
				plate_number=plate_number,
				strike_sample=main_strike_sample,
				analog_rate=analog_rate,
				start_time=start_time,
				point_rate=point_rate,
			)

		for strike_sample, off_sample in pairs:
			leg_context = detect_leg_context_from_plate(
				c3d_obj=c3d_obj,
				plate_number=plate_number,
				strike_sample=strike_sample,
				analog_rate=analog_rate,
				start_time=start_time,
				point_rate=point_rate,
			)

			strike_time = start_time + strike_sample / analog_rate
			off_time = start_time + off_sample / analog_rate
			detected_events.append((strike_time, "Foot Strike", leg_context))
			detected_events.append((off_time, "Foot Off", leg_context))
			n_force_strikes += 1
			n_force_offs += 1

			heel_label, toe_label = resolve_leg_marker_labels(c3d_obj, leg_context)
			following_strike = detect_following_foot_strike_oconnor(
				c3d_obj=c3d_obj,
				heel_label=heel_label,
				toe_label=toe_label,
				toe_off_time=off_time,
				start_time=start_time,
				point_rate=point_rate,
			)
			if following_strike is not None:
				detected_events.append((following_strike, "Foot Strike", leg_context))

	if not detected_events:
		raise RuntimeError("No gait events detected from force plates.")

	detected_events = deduplicate_events(detected_events)
	n_left_events = sum(1 for _, _, c in detected_events if c == "Left")
	n_right_events = sum(1 for _, _, c in detected_events if c == "Right")

	if main_plate_leg is None:
		# Fallback if main plate has no stance in this trial.
		main_plate_leg = "Left" if n_left_events >= n_right_events else "Right"

	incomplete_cycle = not has_complete_gait_cycle(detected_events, main_plate_leg)

	existing_events = load_existing_events(c3d_obj)
	write_events(c3d_obj, detected_events)

	output_path = make_output_path(file_path, main_plate_leg, incomplete_cycle=incomplete_cycle)
	c3d_obj.write(output_path)

	return {
		"input": file_path,
		"output": output_path,
		"n_existing_events": len(existing_events),
		"n_detected_events": len(detected_events),
		"n_total_events": len(existing_events) + len(detected_events),
		"n_force_strikes": n_force_strikes,
		"n_force_offs": n_force_offs,
		"n_left_events": n_left_events,
		"n_right_events": n_right_events,
		"main_plate_leg": main_plate_leg,
		"incomplete_cycle": incomplete_cycle,
	}


def main():
	"""Run the full script from user file selection to final report."""
	root = tk.Tk()
	root.withdraw()

	files = select_c3d_files(root)
	if not files:
		messagebox.showinfo("No Files", "No C3D files selected. Exiting.")
		root.destroy()
		return

	first_c3d = ezc3d.c3d(files[0])
	plate_count = get_plate_count(first_c3d)
	if plate_count < 1:
		messagebox.showerror("Error", "No force plates found in selected C3D file.")
		root.destroy()
		return

	main_plate_number = ask_plate_index(root, 1, plate_count)
	if main_plate_number is None:
		messagebox.showinfo("Cancelled", "Main force plate selection cancelled. Exiting.")
		root.destroy()
		return

	results = []
	errors = []

	for file_path in files:
		try:
			result = process_file(
				file_path=file_path,
				main_plate_number=main_plate_number,
				strike_threshold_n=THRESHOLD_STRIKE_N,
				off_threshold_n=THRESHOLD_OFF_N,
			)
			results.append(result)
		except Exception as exc:
			errors.append((file_path, str(exc)))

	root.destroy()

	print("=" * 80)
	print("GAIT EVENT DETECTION FINISHED")
	print("=" * 80)
	print(f"Foot Strike threshold: {THRESHOLD_STRIKE_N:.1f} N")
	print(f"Foot Off threshold: {THRESHOLD_OFF_N:.1f} N")
	print(f"Main force plate for naming: {main_plate_number}")
	print("Force plates: all available plates are processed")
	print("Output folder name: added_gait_events")
	print("Incomplete cycles are saved in: added_gait_events/incomplete_gait_cycle")
	print()

	if results:
		print("Processed files:")
		for r in results:
			print(f"- Input:  {r['input']}")
			print(f"  Output: {r['output']}")
			print(f"  Naming side from main plate: {r['main_plate_leg']}")
			print(f"  Complete gait cycle: {not r['incomplete_cycle']}")
			print(
				f"  Added events: {r['n_detected_events']} "
				f"(force strikes={r['n_force_strikes']}, force offs={r['n_force_offs']})"
			)
			print(
				f"  Side split: left={r['n_left_events']}, "
				f"right={r['n_right_events']}"
			)
			print(f"  Total events in file: {r['n_total_events']}")
			print()

	if errors:
		print("Files with errors:")
		for file_path, err in errors:
			print(f"- {file_path}")
			print(f"  Error: {err}")


if __name__ == "__main__":
	main()
