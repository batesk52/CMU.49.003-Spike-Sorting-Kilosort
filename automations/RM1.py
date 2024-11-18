# Imports
from pathlib import Path
from tqdm import tqdm
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
import sys
sys.path.insert(1, '../automations')
import RM1_pipeline

from probeinterface import write_prb
# Download channel maps for default probes
from kilosort.utils import download_probes
from probeinterface.plotting import plot_probe, plot_probe_group
from probeinterface import Probe, get_probe, generate_linear_probe

from spikeinterface.extractors import read_intan

from kilosort import run_kilosort
from kilosort import io