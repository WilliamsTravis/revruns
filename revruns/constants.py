# -*- coding: utf-8 -*-
"""Constants for revruns.

Created on Wed Jun 24 20:52:25 2020

@author: twillia2
"""
import os

from osgeo import gdal


# GDAL
GDAL_TYPES = {"GDT_Byte": "Eight bit unsigned integer",
              "GDT_CFloat32": "Complex Float32",
              "GDT_CFloat64": "Complex Float64",
              "GDT_CInt16": "Complex Int16",
              "GDT_CInt32": "Complex Int32",
              "GDT_Float32": "Thirty two bit floating point",
              "GDT_Float64": "Sixty four bit floating point",
              "GDT_Int16": "Sixteen bit signed integer",
              "GDT_Int32": "Thirty two bit signed integer",
              "GDT_UInt16": "Sixteen bit unsigned integer",
              "GDT_UInt32": "Thirty two bit unsigned integer",
              "GDT_Unknown": "Unknown or unspecified type"}

GDAL_TYPEMAP = {"byte": gdal.GDT_Byte,
                "cfloat32": gdal.GDT_CFloat32,
                "cfloat64": gdal.GDT_CFloat64,
                "cint16": gdal.GDT_CInt16,
                "cint32": gdal.GDT_CInt32,
                "float32": gdal.GDT_Float32,
                "float64": gdal.GDT_Float64,
                "int16": gdal.GDT_Int16,
                "int32": gdal.GDT_Int32,
                "uint16": gdal.GDT_UInt16,
                "uint32": gdal.GDT_UInt32,
                "unknown": gdal.GDT_Unknown}

# For filtering COUNS
CONUS_FIPS = ["54", "12", "17", "27", "24", "44", "16", "33", "37", "50", "09",
              "10", "35", "06", "34", "55", "41", "31", "42", "53", "22", "13",
              "01", "49", "39", "48", "08", "45", "40", "47", "56", "38", "21",
              "23", "36", "32", "26", "05", "28", "29", "30", "20", "18", "46",
              "25", "51", "11", "19", "04"]

# For package data
ROOT = os.path.abspath(os.path.dirname(__file__))


# For checking if a requested output requires economic treatment.
ECON_MODULES = ["flip_actual_irr",
                "lcoe_nom",
                "lcoe_real",
                "ppa_price",
                "project_return_aftertax_npv"]

# Checks for reasonable model output value ranges. No scaling factors here.
VARIABLE_CHECKS = {
        "poa": (0, 1000),  # 1,000 MW m-2
        "cf_mean": (0, 240),  # 24 %
        "cf_profile": (0, 990),  # 99 %
        "ghi_mean": (0, 1000),
        "lcoe_fcr": (0, 1000)
        }

# Resource data set dimensions. Just the number of grid points for the moment.
RESOURCE_DIMS = {
    "nsrdb_v3": 2018392,
    "wind_conus_v1": 2488136,
    "wind_canada_v1": 2894781,
    "wind_canada_v1bc": 2894781,
    "wind_mexico_v1": 1736130,
    "wind_conus_v1_1": 2488136,
    "wind_canada_v1_1": 289478,
    "wind_canada_v1_1bc": 289478,
    "wind_mexico_v1_1": 1736130
}

# The Eagle HPC path to each resource data set. Brackets indicate years.
RESOURCE_DATASETS = {
        "nsrdb_v3": "/datasets/NSRDB/v3/nsrdb_{}.h5",
        "nsrdb_india": "/datasets/NSRDB/india/nsrdb_india_{}.h5",
        "wtk_conus_v1": "/datasets/WIND/conus/v1.0.0/wtk_conus_{}.h5",
        "wtk_canada_v1": "/datasets/WIND/canada/v1.0.0/wtk_canada_{}.h5",
        "wtk_canada_v1bc": "/datasets/WIND/canada/v1.0.0bc/wtk_canada_{}.h5",
        "wtk_mexico_v1": "/datasets/WIND/mexico/v1.0.0/wtk_mexico_{}.h5",
        "wtk_conus_v1_1": "/datasets/WIND/conus/v1.1.0/wtk_conus_{}.h5",
        "wtk_canada_v1_1": "/datasets/WIND/canada/v1.1.0/wtk_canada_{}.h5",
        "wtk_canada_v1_1bc": ("/datasets/WIND/canada/v1.1.0bc/" +
                                "wtk_canada_{}.h5"),
        "wtk_mexico_v1_1": "/datasets/WIND/mexico/v1.0.0/wtk_mexico_{}.h5"
        }


# The title of each resource data set.
RESOURCE_LABELS = {
        "nsrdb_v3": "National Solar Radiation Database -  v3.0.1",
        "nsrdb_india": "National Solar Radiation Database - India",
        "wtk_conus_v1": ("Wind Integration National Dataset (WIND) " +
                          "Toolkit - CONUS, v1.0.0"),
        "wtk_canada_v1": ("Wind Integration National Dataset (WIND) " +
                           "Toolkit - Canada, v1.0.0"),
        "wtk_canada_v1bc": ("Wind Integration National Dataset (WIND) " +
                             "Toolkit - Canada, v1.1.0"),
        "wtk_mexico_v1": ("Wind Integration National Dataset (WIND) " +
                           "Toolkit - Mexico, v1.0.0"),
        "wtk_conus_v1_1":("Wind Integration National Dataset (WIND) " +
                           "Toolkit - CONUS, v1.1.0"),
        "wtk_canada_v1_1": ("Wind Integration National Dataset (WIND) " +
                             "Toolkit - Canada, v1.1.0"),
        "wtk_canada_v1_1bc": ("Wind Integration National Dataset (WIND) " +
                               "Toolkit - Canada, v1.1.0bc"),
        "wtk_mexico_v1_1": ("Wind Integration National Dataset (WIND) " +
                             "Toolkit - Mexico, v1.0.0"),
        }

# Target geographic coordinate system identifiers.
TARGET_CRS = ["+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ",
              {"init": "epsg:4326"},
              {"type": "EPSG", "properties": {"code": 4326}}]

# Template config files
BATCH_TEMPLATE = {
  "pipeline_config": "./config_pipeline.json",
  "sets": [
    {
      "args": {
        "PLACEHOLDER": "PLACEHOLDER",
        "PLACEHOLDER": "PLACEHOLDER"
      },
      "files": [
        "./sam_configs/default.json"
      ],
      "set_tag": "set1"
    }
  ]
}
BESPOKE_TEMPLATE = {
    "log_directory": "./logs",
    "execution_control": {
        "allocation": "wetosa",
        "feature": "--qos=normal",
        "option": "eagle",
        "walltime": 12,
        "nodes": 100,
        "memory": 79
    },
    "excl_dict":  {
        "PLACEHOLDER": {
            "exclude_values": "PLACEHOLDER"
        }
    },
    "excl_fpath": "/projects/rev/data/exclusions/ATB_Exclusions.h5",
    "log_level": "DEBUG",
    "res_fpath": "/datasets/WIND/conus/v1.0.0/wtk_conus_*.h5",
    "tm_dset": "techmap_wind",
    "project_points": "PLACEHOLDER",
    "sam_files": {
        "onshore": "PLACEHOLDER"
    },
    "capital_cost_function": "PLACEHOLDER",
    "fixed_operating_cost_function": "PLACEHOLDEER",
    "variable_operating_cost_function": "0",
    "objective_function": "PLACEHOLDER",
    "ga_kwargs": {
        "convergence_iters": 100,
        "max_generation": 1000,
        "max_time": 15000
    },
    "output_request": [
        "system_capacity",
        "cf_mean",
        "cf_profile",
        "wake_losses",
        "annual_gross_energy",
        "winddirection",
        "ws_mean",
        "lcoe_fcr",
        "fixed_charge_rate",
        "capital_cost",
        "fixed_operating_cost",
        "variable_operating_cost"
    ],
    "resolution": 128,
    "pre_extract_inclusions": False,
    "wake_loss_multiplier": 1
}

GEN_TEMPLATE = {
    "log_directory": "./logs",
    "execution_control": {
        "allocation": "PLACEHOLDER",
        "feature": "--qos=normal",
        "memory_utilization_limit": 0.4,
        "nodes": 10,
        "option": "eagle",
        "sites_per_worker": 100,
        "walltime": 4.0
    },

    "log_level": "INFO",
    "analysis_years": [
        2007,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013
    ],
    "technology": "PLACEHOLDER",
    "output_request": [
      "cf_mean",
      "cf_profile"
    ],
    "project_points": "PLACEHOLDER",
    "sam_files": {
        "key": "PLACEHOLDER"
    },
    "resource_file": "PLACEHOLDER"
}

OFFSHORE_TEMPLATE = {
  "log_directory": "./logs/",
  "execution_control": {
      "allocation": "PLACEHOLDER",
      "option": "eagle",
      "walltime": 1
  },
  "gen_fpath": "PIPELINE",
  "log_level": "INFO",
  "offshore_fpath": "PLACEHOLDER",
  "offshore_meta_cols": [
      "depth",
      "dist_p_to_a",
      "dist_a_to_s",
      "dist_op_to_s",
      "dist_p_to_s",
      "dist_p_to_s_nolimit",
      "dist_s_to_l",
      "hs_average",
      "fixed_downtime",
      "floating_downtime"
  ],
  "offshore_nrwal_keys": [
      "depth",
      "dist_p_to_a",
      "dist_a_to_s",
      "dist_op_to_s",
      "dist_p_to_s",
      "dist_p_to_s_nolimit",
      "dist_s_to_l",
      "hs_average",
      "fixed_downtime",
      "floating_downtime",
      "capital_cost",
      "fixed_operating_cost",
      "variable_operating_cost",
      "fixed_charge_rate",
      "system_capacity"
  ],
  "project_points": "PLACEHOLDER",
  "sam_files": {
      "fixed": "PLACEHOLDER",
      "floating": "PLACEHOLDER"
  },
  "nrwal_configs": {
      "offshore": "PLACEHOLDER"
  }
}

COLLECT_TEMPLATE = {
    "log_directory": "./logs",
    "execution_control": {
        "allocation": "PLACEHOLDER",
        "feature": "--qos=normal",
        "memory": 90,
        "option": "eagle",
        "walltime": 1.0
    },
    "dsets": [
        "cf_mean",
        "cf_profile",
        "lcoe_fcr",
        "ghi_mean"
    ],
    "file_prefixes": "PIPELINE",
    "log_level": "INFO",
    "project_points": "PLACEHOLDER"
}

ECON_TEMPLATE = {
  "analysis_years": [
    2007,
	  2008,
	  2009,
    2010,
    2011,
  	2012,
  	2013
  ],
  "append": True,
  "cf_file": "PLACEHOLDER",
  "log_directory": "./logs",
  "execution_control": {
    "allocation": "PLACEHOLDER",
    "nodes": "PLACEHOLDER",
    "option": "eagle",
    "memory": 90,
    "memory_utilization_limit": 0.4,
    "nodes": 10,
    "option": "eagle",
    "sites_per_worker": 100,
    "walltime": 3.0
  },
  "log_level": "INFO",
  "output_request": [
    "lcoe_fcr"
  ],
  "project_points": "./project_points.csv",
  "sam_files": "PLACEHOLDER",
  "technology": "PLACEHOLDER",
  "site_data": "PLACEHOLDER"
}


MULTIYEAR_TEMPLATE = {
  "log_directory": "./logs",
  "execution_control": {
    "allocation": "PLACEHOLDER",
    "feature": "--qos=normal",
    "memory": 179,
    "option": "eagle",
    "walltime": 2.0
  },
  "groups": {
    "none": {
      "dsets": "PLACEHOLDER",
      "source_dir": "./",
      "source_files": "PIPELINE",
      "source_prefix": ""
    }
  },
  "log_level": "INFO"
}

AGGREGATION_TEMPLATE = {
  "cf_dset": "cf_mean-means",
  "data_layers": {
    "slope": {
      "dset": "srtm_slope",
      "method": "mean"
    },
    "model_region": {
      "dset": "reeds_regions",
      "method": "mode"
    }
  },
  "log_directory": "./logs",
  "excl_dict": {
    "PLACEHOLDER": {
      "exclude_values": "PLACEHOLDER"
    },
  },
  "excl_fpath": "PLACEHOLDER",
  "execution_control": {
    "allocation": "PLACEHOLDER",
    "feature": "--qos=normal",
    "memory": 90,
    "nodes": 1,
    "option": "eagle",
    "walltime": 1.0
  },
  "gen_fpath": "PIPELINE",
  "lcoe_dset": "PIPELINE",
  "power_density": "PLACEHOLDER",
  "res_class_bins": "PLACEHOLDER",
  "res_class_dset": "PLACEHOLDER",
  "res_fpath": "PLACHOLDER",
  "resolution": "PLACEHOLDER",
  "tm_dset": "PLACHOLDER"
}

SUPPLY_CURVE_TEMPLATE_OLD = {
  "avail_cap_frac": "PLACEHOLDER",
  "log_directory": "./logs",
  "execution_control": {
    "allocation": "PLACEHOLDER",
    "feature": "--qos=normal",
    "memory": 90,
    "nodes": 4,
    "option": "eagle",
    "walltime": 1.0
  },
  "fixed_charge_rate": "PLACEHOLDER",
  "sc_features": "/projects/rev/data/transmission/build/multipliers_128.csv",
  "sc_points": "PIPELINE",
  "simple": True,
  "trans_table": "/projects/rev/data/transmission/build/connections_128.csv",
  "transmission_costs": {
    "center_tie_in_cost": "PLACEHOLDER",
    "line_cost": "PLACEHOLDER",
    "line_tie_in_cost": "PLACEHOLDER",
    "sink_tie_in_cost": "PLACEHOLDER",
    "station_tie_in_cost": "PLACEHOLDER"
  }
}

SUPPLY_CURVE_TEMPLATE_LC = {
    "log_directory": "./logs",
    "execution_control": {
        "allocation": "PLACEHOLDER",
        "feature": "--qos=normal",
        "option": "eagle",
        "memory": 90,
        "nodes": 4,
        "walltime": 2
    },
    "fixed_charge_rate": "PLACEHOLDER",
    "sc_points": "PIPELINE",
    "simple": True,  # Must be true
    "trans_table": [
        "/shared-projects/rev/exclusions/least_cost_xmission/100MW_costs_128.csv",
        "/shared-projects/rev/exclusions/least_cost_xmission/200MW_costs_128.csv",
        "/shared-projects/rev/exclusions/least_cost_xmission/400MW_costs_128.csv",
        "/shared-projects/rev/exclusions/least_cost_xmission/1000MW_costs_128.csv"  # For Large ReEDS solar runs
    ]
}

REP_PROFILES_TEMPLATE = {
  "cf_dset": "cf_profile-{}",
  "log_directory": "./logs",
  "err_method": "rmse",
  "execution_control": {
    "allocation": "PLACEHOLDER",
    "feature": "--qos=normal",
    "memory": 179,
    "nodes": 1,
    "option": "eagle",
    "sites_per_worker": 100,
    "walltime": 1.0
  },
  "gen_fpath": "PIPELINE",
  "analysis_years": [
    2012
  ],
  "reg_cols": "sc_point_gid",
  "log_level": "DEBUG",
  "rev_summary": "PIPELINE",
  "aggregate_profiles": True
}

PIPELINE_TEMPLATE = {
    "logging": {
        "log_file": None,
        "log_level": "INFO"
    },
    "pipeline": [
        {"generation": "./config_gen.json"},
        {"collect": "./config_collect.json"},
        {"econ": "./config_econ.json"},
        {"multi-year": "./config_multi-year.json"},
        {"supply-curve-aggregation": "./config_aggregation.json"},
        {"supply-curve": "./config_supply-curve.json"},
        {"rep-profiles": "./config_rep-profiles.json"}
    ]
}

TEMPLATES = {
    "gen": GEN_TEMPLATE,
    "bsp": BESPOKE_TEMPLATE,
    "off": OFFSHORE_TEMPLATE,
    "co": COLLECT_TEMPLATE,
    "econ": ECON_TEMPLATE,
    "my": MULTIYEAR_TEMPLATE,
    "ag": AGGREGATION_TEMPLATE,
    "sc_old": SUPPLY_CURVE_TEMPLATE_OLD,
    "sc_lc": SUPPLY_CURVE_TEMPLATE_LC,
    "rp": REP_PROFILES_TEMPLATE,
    "ba": BATCH_TEMPLATE,
    "pi": PIPELINE_TEMPLATE
}

# Default SAM model parameters
SOLAR_SAM_PARAMS = {
    "azimuth": "PLACEHOLDER",
    "array_type": "PLACEHOLDER",
    "capital_cost": "PLACEHOLDER",
    "clearsky": "PLACEHOLDER",
    "compute_module": "PLACEHOLDER",
    "dc_ac_ratio": "PLACEHOLDER",
    "fixed_charge_rate": "PLACEHOLDER",
    "fixed_operating_cost": "PLACEHOLDER",
    "gcr": "PLACEHOLDER",
    "inv_eff": "PLACEHOLDER",
    "losses": "PLACEHOLDER",
    "module_type": "PLACEHOLDER",
    "system_capacity": "PLACEHOLDER",
    "tilt": "PLACEHOLDER",
    "variable_operating_cost": "PLACEHOLDER"
}

WIND_SAM_PARAMS = {
        "adjust:constant": 0,
        "capital_cost" : "PLACEHOLDER",
        "fixed_operating_cost" : "PLACEHOLDER",
        "fixed_charge_rate": "PLACEHOLDER",
        "icing_cutoff_temp": "PLACEHOLDER",
        "icing_cutoff_rh": "PLACEHOLDER",
        "low_temp_cutoff": "PLACEHOLDER",
        "system_capacity": "PLACEHOLDER",
        "variable_operating_cost": 0,
        "turb_generic_loss": 16.7,
        "wind_farm_wake_model": 0,
        "wind_farm_xCoordinates": [0],
        "wind_farm_yCoordinates": [0],
        "wind_resource_model_choice": 0,
        "wind_resource_shear": 0.14,
        "wind_resource_turbulence_coeff": 0.1,
        "wind_turbine_cutin": "PLACEHOLDER",  # Isn't this inferred in the pc?
        "wind_turbine_hub_ht": "PLACEHOLDER",
        "wind_turbine_powercurve_powerout": "PLACEHOLDER",
        "wind_turbine_powercurve_windspeeds": "PLACEHOLDER",
        "wind_turbine_rotor_diameter": "PLACEHOLDER"
}

SAM_TEMPLATES = {
    "pvwattsv5": SOLAR_SAM_PARAMS,
    "pvwattsv7": SOLAR_SAM_PARAMS,
    "windpower": WIND_SAM_PARAMS
}

SLURM_TEMPLATE = (
"""#!/bin/bash

#SBATCH --account=PLACEHOLDER
#SBATCH --time=1:00:00
#SBATCH -o PLACEHOLDER.o
#SBATCH -e PLACEHOLDER.e
#SBATCH --job-name=<PLACEHOLDER>
#SBATCH --nodes=1
#SBATCH --mail-user=PLACEHOLDER
#SBATCH --mem=79000

echo Running on: $HOSTNAME, Machine Type: $MACHTYPE
echo CPU: $(cat /proc/cpuinfo | grep "model name" -m 1 | cut -d:  -f2)
echo RAM: $(free -h | grep  "Mem:" | cut -c16-21)

source ~/.bashrc
module load conda
conda activate /path/to/env/

python script.py
"""
)