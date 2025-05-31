##############################################
# IMPORT LIBRARIES ###########################
##############################################

"""
Libraries and packages used in this script and Versions Info for tools, libraries, and packages used in this script.
"""
import os
import sys               # Version: Python Standard Library
import subprocess        # Version: Python Standard Library
import argparse          # Version: Python Standard Library
import json              # Version: Python Standard Library

###############################################
# CONSTANTS & PARAMETERS ######################
###############################################

"""
Constants and parameters used in this script.
"""
# Default JSON config path
CONFIG_PATH = 'config_riscv.json'

# Script filenames (relative to this script)
EXPORT_MLP_SCRIPT    = 'onnx_any_ep_only_export.py'
BENCHMARK_SCRIPT     = 'onnx-mlir_test_any_ep_riscv.py'

##############################################
# FUNCTION DEFINITIONS #######################
##############################################

def run_step(name, command, cwd=None):
    """
    Run a subprocess command as part of the pipeline.

    Args:
        name (str): Name of the pipeline step for logging.
        command (list): List of strings representing the command and args.
        cwd (str, optional): Working directory for the command.

    Raises:
        SystemExit: Exits if a step fails with non-zero return code.
    """
    print(f"\n=== Step: {name} ===")
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=cwd or os.getcwd())
    if result.returncode != 0:
        print(f"❌ {name} failed (exit code {result.returncode}). Exiting pipeline.")
        sys.exit(result.returncode)
    print(f"✅ {name} completed successfully.")

##############################################
# MAIN PROGRAM ################################
##############################################

if __name__ == '__main__':
    #########################################
    # PARSE COMMAND-LINE ARGUMENTS ##########
    #########################################
    parser = argparse.ArgumentParser(
        description='End-to-end pipeline: export, compile, and benchmark.'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=CONFIG_PATH,
        help='Path to JSON configuration file.'
    )
    args = parser.parse_args()

    # Validate config file
    if not os.path.isfile(args.config):
        print(f"❌ Config file not found: {args.config}")
        sys.exit(1)

    # Make sure our scripts are present
    for script in [EXPORT_MLP_SCRIPT, BENCHMARK_SCRIPT]:
        if not os.path.exists(script):
            print(f"❌ Required script not found: {script}")
            sys.exit(1)

    #########################################
    # STEP 1: EXPORT MLP ➔ ONNX #############
    #########################################
    run_step(
        'Export MLP to ONNX (vanilla & optimized)',
        [sys.executable, EXPORT_MLP_SCRIPT, '--config', args.config]
    )

    #########################################
    # STEP 2: COMPILE & BENCHMARK WITH MLIR##
    #########################################
    run_step(
        'Compile & benchmark with ONNX-MLIR',
        [sys.executable, BENCHMARK_SCRIPT, '--config', args.config]
    )
