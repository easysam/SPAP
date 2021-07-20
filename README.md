# Simultaneous Demand Prediction and Planning

## Dependencies
Python packages: `Pytorch`, `scikit-learn`, `Pandas`, `Numpy`, `PyYAML`

## Data
POI: `data/poi`<br>
Road network: `data/roadnet`<br>
Transportation: `data/transportation`<br>
Station profile: `data/station_list`<br>
Charging records: `data/station`<br>
Station profile and charging records can be collected in _Star Charge_ APP.

## Experiments

1. Evaluations of charging demand prediction<br>
   `python model/run.py --source SOURCE_CITY --target TARGET_CITY --model MODEL_NAME`
2. Evaluations based on real plans<br>
    1. Even, CG, Real, TIO: `python real_world.py`
    2. Park: `python real_world_parking.py`
    3. Pop: `python real_world_population.py`
3. Evaluations with varied budgets<br>
    `python evaluation_varied_budget.py`
4. Evaluations with optimal solution<br>
    `python optimal_comparison_BF_tio.py`
