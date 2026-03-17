BlackhawksApp
=============

Small Streamlit scaffold to visualize JEEDS intermediate estimates and
Blackhawks angular heatmaps.

Implemented so far
------------------

- `BlackhawksApp/data_io.py`: centralized app-facing data access.
	- `get_players()`
	- `get_seasons(player_id)`
	- `get_intermediate_csvs(player_id)`
	- `load_estimates(csv_path)`
	- `load_heatmaps(player_id, seasons=..., tag=...)`
	- `get_heatmap_for_shot(...)`
- `BlackhawksApp/app.py`: uses `data_io` and Streamlit caching
	(`st.cache_data` and `st.cache_resource`).
- The app now renders from cached in-memory shot blobs (`df` + `shot_maps`)
	to avoid reloading parquet/npz on every plot request.
- `BlackhawksApp/benchmark_loaders.py`: benchmark current parquet+npz loader performance.

Run:

```bash
pip install -r BlackhawksApp/requirements.txt
python -m streamlit run BlackhawksApp/app.py
```

Benchmark loader performance:

```bash
python -m BlackhawksApp.benchmark_loaders --player 950160 --seasons 20242025 --repeats 3
```
