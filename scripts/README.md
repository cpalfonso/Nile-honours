## Scripts

To run one of the models, `cd` into the appropriate directory (i.e. `scripts/{model_name}`), then run `run_models.py`.
Outputs will be placed in the `results/{model_name}` directory.

To visualise the results, either load the `tin.series.xdmf` file using ParaView or run the `visualise_results.py` and `create_animations.sh` scripts.

`visualise_results.py` will create snapshots of elevation/discharge and cumulative erosion/deposition for each time step of the model.
`create_animations.sh` will use these snapshots to create animations showing the evolution of the model through time.
This script requires FFmpeg ([ffmpeg.org](https://ffmpeg.org)).
