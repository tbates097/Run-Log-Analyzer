import os
import sys
import pandas as pd
import plotly.express as px

# Configuration
RUNLOG_PATH = r"Z:\Aerotech USA\03. Aerotech USA\4. Manufacturing\METROLOGY\RunLog.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
STOP_VALUE = "Stopped"

# Expected columns (based on sample testcsv.csv)
COL_ABORTED = "Aborted"
COL_USER = "User Name"
COL_TEST_NAME = "Test Name"
COL_TEST_DATETIME = "Test Date Time"
COL_PART_NUMBER = "Part Number"
COL_STAGE_SERIAL = "Stage Serial Number"


def ensure_output_dir() -> None:
	os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_runlog(csv_path: str) -> pd.DataFrame:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"RunLog.csv not found at: {csv_path}")
	# Use low_memory=False for mixed dtypes
	df = pd.read_csv(csv_path, low_memory=False)
	return df


def normalize_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
	if column not in df.columns:
		raise KeyError(f"Missing expected column: {column}")
	# Parse datetimes
	df[column] = pd.to_datetime(df[column], errors="coerce")
	return df


def filter_stops(df: pd.DataFrame) -> pd.DataFrame:
	missing = [c for c in [COL_ABORTED, COL_USER, COL_TEST_NAME, COL_TEST_DATETIME, COL_PART_NUMBER, COL_STAGE_SERIAL] if c not in df.columns]
	if missing:
		raise KeyError(f"Missing expected columns: {missing}")
	stops = df[df[COL_ABORTED].astype(str).str.strip().str.casefold() == STOP_VALUE.casefold()].copy()
	stops = normalize_datetime(stops, COL_TEST_DATETIME)
	# Keep only needed columns
	stops = stops[[COL_USER, COL_TEST_NAME, COL_TEST_DATETIME, COL_PART_NUMBER, COL_STAGE_SERIAL]]
	stops = stops.rename(columns={
		COL_USER: "user_name",
		COL_TEST_NAME: "test_name",
		COL_TEST_DATETIME: "test_datetime",
		COL_PART_NUMBER: "part_number",
		COL_STAGE_SERIAL: "stage_serial",
	})
	# Normalize user names (strip whitespace and convert to lowercase)
	stops["user_name"] = stops["user_name"].astype(str).str.strip().str.lower()
	# Drop rows without a parsed datetime
	stops = stops.dropna(subset=["test_datetime"]).sort_values("test_datetime").reset_index(drop=True)
	return stops


def save_stops_csv(stops: pd.DataFrame) -> str:
	ensure_output_dir()
	out_path = os.path.join(OUTPUT_DIR, "stopped_events.csv")
	stops.to_csv(out_path, index=False)
	return out_path


def save_plotly_html(fig, filename: str) -> str:
	ensure_output_dir()
	path = os.path.join(OUTPUT_DIR, filename)
	fig.write_html(path, include_plotlyjs="cdn", full_html=True)
	return path


def plot_bar_series_html(series: pd.Series, title: str, x_label: str, y_label: str, filename: str, orientation: str = "h") -> str:
	data = series.sort_values(ascending=False).reset_index()
	data.columns = ["category", "count"]
	if orientation == "h":
		fig = px.bar(data, x="count", y="category", orientation="h", text="count", title=title, labels={"count": y_label, "category": x_label})
		fig.update_layout(yaxis={"categoryorder": "total ascending"})
	else:
		fig = px.bar(data, x="category", y="count", orientation="v", text="count", title=title, labels={"count": y_label, "category": x_label})
		fig.update_layout(xaxis={"categoryorder": "total descending"})
	fig.update_traces(textposition="auto")
	fig.update_layout(margin=dict(l=60, r=20, t=60, b=60), height=700)
	return save_plotly_html(fig, filename)


def plot_stops_per_user(stops: pd.DataFrame) -> str:
	counts = stops.groupby("user_name").size()
	return plot_bar_series_html(counts, "Stops per User", "User", "Stops", "stops_per_user.html", orientation="h")


def plot_stops_per_test(stops: pd.DataFrame) -> str:
	counts = stops.groupby("test_name").size()
	return plot_bar_series_html(counts, "Stops per Test Name", "Test Name", "Stops", "stops_per_test.html", orientation="h")


def plot_stops_per_part(stops: pd.DataFrame) -> str:
	counts = stops.groupby("part_number").size()
	return plot_bar_series_html(counts, "Stops per Part Number", "Part Number", "Stops", "stops_per_part.html", orientation="h")


def plot_stops_per_stage(stops: pd.DataFrame) -> str:
	# Group by stage_serial and aggregate user names and part numbers for hover data
	stage_data = stops.groupby("stage_serial").agg({
		"user_name": lambda x: ", ".join(sorted(set(x))),
		"part_number": lambda x: ", ".join(sorted(set(x))),
		"stage_serial": "size"
	}).rename(columns={"stage_serial": "count"})
	
	# Create custom hover text
	stage_data["hover_text"] = stage_data.apply(
		lambda row: f"Stage: {row.name}<br>Stops: {row['count']}<br>Users: {row['user_name']}<br>Parts: {row['part_number']}", 
		axis=1
	)
	
	# Create the plot with custom hover data
	fig = px.bar(
		stage_data.reset_index(), 
		x="count", 
		y="stage_serial", 
		orientation="h", 
		text="count", 
		title="Stops per Stage Serial Number",
		labels={"count": "Stops", "stage_serial": "Stage Serial Number"},
		custom_data=["hover_text"]
	)
	fig.update_traces(
		hovertemplate="%{customdata[0]}<extra></extra>",
		textposition="auto"
	)
	fig.update_layout(
		yaxis={"categoryorder": "total ascending"},
		margin=dict(l=60, r=20, t=60, b=60), 
		height=700
	)
	return save_plotly_html(fig, "stops_per_stage.html")


def analyze_consecutive_stops_by_test_stage(stops: pd.DataFrame) -> pd.DataFrame:
	"""
	Sequences: consecutive stops grouped by identical (test_name, stage_serial), ordered by time.
	No time-window or user constraint.
	"""
	if stops.empty:
		return stops.assign(sequence_id=pd.Series(dtype=int), sequence_index=pd.Series(dtype=int))

	stops = stops.sort_values(["test_name", "stage_serial", "test_datetime"]).reset_index(drop=True).copy()
	sequence_ids = []
	sequence_index = []
	current_seq = 0
	last_key = (None, None)
	last_seq_index_for_key: dict[tuple[str, str], int] = {}

	for _, row in stops.iterrows():
		key = (str(row["test_name"]), str(row["stage_serial"]))
		if key != last_key:
			current_seq += 1
			sequence_ids.append(current_seq)
			sequence_index.append(1)
			last_seq_index_for_key[key] = 1
			last_key = key
		else:
			sequence_ids.append(current_seq)
			last_seq_index_for_key[key] += 1
			sequence_index.append(last_seq_index_for_key[key])

	stops["sequence_id"] = sequence_ids
	stops["sequence_index"] = sequence_index
	return stops


def summarize_sequences_by_test_stage(stops_with_seq: pd.DataFrame) -> pd.DataFrame:
	if "sequence_id" not in stops_with_seq.columns:
		return pd.DataFrame()
	summary = (
		stops_with_seq.groupby("sequence_id")
			.agg(
				start_time=("test_datetime", "min"),
				end_time=("test_datetime", "max"),
				num_stops=("sequence_index", "max"),
				user_name=("user_name", "first"),
				test_name=("test_name", "first"),
				part_number=("part_number", "first"),
				stage_serial=("stage_serial", "first"),
			)
			.reset_index()
	)
	return summary


def plot_sequence_histogram(seq_summary: pd.DataFrame) -> str:
	if seq_summary.empty:
		return ""
	fig = px.histogram(seq_summary, x="num_stops", nbins=int(seq_summary["num_stops"].max()), title="Stops per (Test Name, Stage Serial) Sequence", labels={"num_stops": "Stops in Sequence"})
	fig.update_layout(margin=dict(l=60, r=20, t=60, b=60), height=500, yaxis_title="Count of Sequences")
	return save_plotly_html(fig, "consecutive_stops_hist.html")


def main() -> int:
	try:
		df = read_runlog(RUNLOG_PATH)
		stops = filter_stops(df)
		stops_csv = save_stops_csv(stops)

		# Visualizations (Plotly HTML)
		user_plot = plot_stops_per_user(stops) if not stops.empty else ""
		test_plot = plot_stops_per_test(stops) if not stops.empty else ""
		part_plot = plot_stops_per_part(stops) if not stops.empty else ""
		stage_plot = plot_stops_per_stage(stops) if not stops.empty else ""

		# Sequence analysis by (Test Name, Stage Serial Number)
		stops_seq = analyze_consecutive_stops_by_test_stage(stops)
		seq_summary = summarize_sequences_by_test_stage(stops_seq)
		seq_csv = os.path.join(OUTPUT_DIR, "consecutive_stops_sequences.csv")
		seq_summary.to_csv(seq_csv, index=False)
		seq_hist = plot_sequence_histogram(seq_summary)

		print("Stopped events saved to:", stops_csv)
		if user_plot: print("HTML saved:", user_plot)
		if test_plot: print("HTML saved:", test_plot)
		if part_plot: print("HTML saved:", part_plot)
		if stage_plot: print("HTML saved:", stage_plot)
		print("Sequences (Test, Stage) saved to:", seq_csv)
		if seq_hist: print("HTML saved:", seq_hist)
		return 0
	except Exception as exc:
		print("Error:", exc)
		return 1


if __name__ == "__main__":
	sys.exit(main())
