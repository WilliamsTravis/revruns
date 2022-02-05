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

GEN_TEMPLATE = {
    "directories": {
        "log_directory": "./logs",
        "output_directory": "./"
    },
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
    "analysis_years": "PLACEHOLDER",
    "technology": "PLACEHOLDER",
    "output_request": "PLACEHOLDER",
    "project_points": "PLACEHOLDER",
    "sam_files": {
        "key": "PLACEHOLDER"
    },
    "resource_file": "PLACEHOLDER"
}

OFFSHORE_TEMPLATE = {
  "directories": {
      "log_directory": "./logs/",
      "output_directory": "./"
  },
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
    "directories": {
        "collect_directory": "PIPELINE",
        "log_directory": "./logs",
        "output_directory": "./"
    },
    "execution_control": {
        "allocation": "PLACEHOLDER",
        "feature": "--qos=normal",
        "memory": 90,
        "option": "eagle",
        "walltime": 1.0
    },
    "dsets": "PLACEHOLDER",
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
  "directories": {
    "log_directory": "./logs",
    "output_directory": "./"
  },
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
  "directories": {
    "log_directory": "./logs",
    "output_directory": "./"
  },
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
  "directories": {
    "log_directory": "./logs",
    "output_directory": "./"
  },
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

SUPPLYCURVE_TEMPLATE = {
  "avail_cap_frac": "PLACEHOLDER",
  "directories": {
    "log_directory": "./logs",
    "output_directory": "./"
  },
  "execution_control": {
    "allocation": "PLACEHOLDER",
    "feature": "--qos=normal",
    "memory": 90,
    "nodes": 4,
    "option": "eagle",
    "walltime": 1.0
  },
  "fixed_charge_rate": "PLACEHOLDER",
  "sc_features": ("/projects/rev/data/transmission/" +
                  "conus_pv_tline_multipliers.csv"),
  "sc_points": "PIPELINE",
  "simple": False,
  "trans_table": ("/projects/rev/data/transmission/" +
                  "conus_trans_lines_cache_offsh_064_sj_infsink.csv"),
  "transmission_costs": {
    "center_tie_in_cost": "PLACEHOLDER",
    "line_cost": "PLACEHOLDER",
    "line_tie_in_cost": "PLACEHOLDER",
    "sink_tie_in_cost": "PLACEHOLDER",
    "station_tie_in_cost": "PLACEHOLDER"
  }
}

REPPROFILES_TEMPLATE = {
  "cf_dset": "cf_profile-{}",
  "directories": {
    "log_directory": "./logs",
    "output_directory": "./"
  },
  "err_method": "rmse",
  "execution_control": {
    "allocation": "PLACEHOLDER",
    "feature": "--qos=normal",
    "memory": 179,
    "nodes": 10,
    "option": "eagle",
    "sites_per_worker": 100,
    "walltime": 2.0
  },
  "gen_fpath": "PIPELINE",
  "n_profiles": 1,
  "analysis_years": "PLACEHOLDER",
  "log_level": "INFO",
  "reg_cols": [
    "model_region",
    "res_class"
  ],
  "rep_method": "meanoid",
  "rev_summary": "PIPELINE"
}

PIPELINE_TEMPLATE = {
    "logging": {
        "log_file": None,
        "log_level": "INFO"
    },
    "pipeline": []
}

TEMPLATES = {
    "gen": GEN_TEMPLATE,
    "off": OFFSHORE_TEMPLATE,
    "co": COLLECT_TEMPLATE,
    "econ": ECON_TEMPLATE,
    "my": MULTIYEAR_TEMPLATE,
    "ag": AGGREGATION_TEMPLATE,
    "sc": SUPPLYCURVE_TEMPLATE,
    "rp": REPPROFILES_TEMPLATE,
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