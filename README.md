# revruns
Examining a way to streamline the configuration and execution of the Renewable Energy Technical Potential Model (reV).

### Install
`pip install git+ssh://git@github.com/NREL/reV.git`\
or\
`pip install git+https://github.com/NREL/reV.git`\
and\
`pip install -e .`

##Examples:
 # Import the Config class from revruns, this will contain all of the configuration information and create the configuration needed to run the reV model.
 
```python
from revruns import Config
```

# Create a configuration object
```python
cnfg = Config()
```

# If you don't want to run reV for all available points (and there are millions), use one of revrun's point object generators. Here, we are using `box_points` to create a data frame of point coordinates and grid ids associated with the
NSRDB V3.0.1 dataset.

# Store top level parameters. These are parameters that will be shared by all model runs.

# Store the individual SAM configuration parameters and generate seperate configuration files for each.

# Generate all of the required configuration files

# Run reV in the command line
`reV -c config_batch.json batch`


### Things to do:
- include options for configuration keys
