# revruns
Examining a way to streamline the configuration and execution of the Renewable Energy Technical Potential Model (reV).

### Install
`pip install git+ssh://git@github.com/NREL/reV.git`\
or\
`pip install git+https://github.com/NREL/reV.git`\
and\
`pip install -e .`

## Examples:
 #### Import the Config class from revruns, this will contain all of the configuration information and create the configuration needed to run the reV model.
 
```python
from revruns import Config
```

#### Create a configuration object
```python
cnfg = Config()
```

#### If you don't want to run reV for all available points (and there are millions), use one of revrun's point object generators. Here, we are using a set of bounding box coordinates and the function `box_points` to create a data frame of point coordinates and grid ids associated with the NSRDB V3.0.1 dataset.
```python
from revruns import box_points

# Points from a bounding box around Denver's portion of the Front Range
bbox = [-105.352679491, 39.4595438351, -104.9022400379, 40.3518303006]
points = box_points(bbox, resource="nsrdb_v3")
```

#### Set the years to be run by creating a list.
```python
# Set years explicitly
years = [y for y in range(2015, 2019)]
```

#### Store top level parameters. These are parameters that will be shared by all model runs.
```python
# Set common parameters
cnfg.top_params["allocation"] = "rev"
cnfg.top_params["logdir"] = "./"
cnfg.top_params["keep_chunks"] = True
cnfg.top_params["memory"] = 192
cnfg.top_params["nodes"] = 25
cnfg.top_params["outdir"] = "./"
cnfg.top_params["outputs"] = ["cf_profile", "cf_mean", "poa"]
cnfg.top_params["resource"] = "nsrdb_v3"
cnfg.top_params["set_tag"] = "nsrdb"
cnfg.top_params["walltime"] = 1.0
cnfg.top_params["years"] = years
cnfg.points = points
cnfg.sam_params["dc_ac_ratio"] = 1.1
cnfg.sam_params["system_capacity"] = 5
```

#### Store the individual Systems Advisor Model (SAM) configuration parameters and generate seperate configuration files for each. SAM is responsible for simulating power generation technologies, fixed and tracking photo-voltaic solar panels in this case.
```python
# SAM Config #1
cnfg.sam_params["array_type"] = 0
cnfg.sam_params["tilt"] = "latitude"
sam_config = cnfg.config_sam(jobname="fixed")

# SAM Config #2
cnfg.sam_params["array_type"] = 2
cnfg.sam_params["tilt"] = 0
sam_config = cnfg.config_sam(jobname="tracking")
```


#### Generate all of the required configuration files
```python
# And this should trigger all of the other configuration files
gen_config = cnfg.config_all(excl_pos_lon=True)
```

#### Run reV in the command line.
`reV -c config_batch.json batch`


### Things to do:
- include options for configuration keys
