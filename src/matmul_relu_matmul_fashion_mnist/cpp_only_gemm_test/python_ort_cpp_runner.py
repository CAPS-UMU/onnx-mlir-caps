#!/usr/bin/env python3

# WORKFLOW: All programs OK

##############################################
# IMPORT LIBRARIES ###########################
##############################################
"""
Libraries and packages used in this script and Versions Info for tools, libraries, and packages used in this script.
"""
import subprocess # Python Standard Library
import sys        # Python Standard Library

###############################################
# CONSTANTS & PARAMETERS ######################
###############################################
"""
Constants and parameters used in this script.
"""
PYTHON_EXECUTABLE = "python3"
EXPORT_SCRIPT_NAME = "export_gemm.py"
ONNX_MODEL_FILE = "gemm.onnx"
C_INCLUDE_FILE = "gemm_model.inc"
XXD_EXECUTABLE = "xxd"
XXD_ARGS = "-i"
CHMOD_EXECUTABLE = "chmod"
CHMOD_ARGS = "+x"
CPP_COMPILE_SCRIPT = "./c++_compile_script.sh"
CPP_EXECUTABLE_NAME = "only_gemm_inmem_onnx_model"

##############################################
# FUNCTION DEFINITIONS #######################
##############################################
"""
Functions must adhere to the following structure:

1) Use docstrings to describe the function's purpose, parameters, and return values.
2) Be named according to their functionality.
"""

def run_subprocess_command(cmd, **kwargs):
    """
    Executes a subprocess command and prints it before execution.

    Args:
        cmd (list): A list of strings representing the command and its arguments.
        **kwargs: Additional keyword arguments to pass to subprocess.run.

    Raises:
        subprocess.CalledProcessError: If the command returns a non-zero exit code.
    """
    print(f"> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)

def main_execution_flow():
    """
    Main execution flow of the script.
    Exports an ONNX model, embeds it as a C include, and then compiles and runs a C++ example.
    """
    #########################################
    # EXPORT ONNX MODEL #####################
    #########################################
    run_subprocess_command([PYTHON_EXECUTABLE, EXPORT_SCRIPT_NAME])

    #########################################
    # EMBED MODEL AS C INCLUDE ##############
    #########################################
    with open(C_INCLUDE_FILE, "wb") as out_file:
        run_subprocess_command([XXD_EXECUTABLE, XXD_ARGS, ONNX_MODEL_FILE], stdout=out_file)

    #########################################
    # COMPILE AND RUN C++ EXAMPLE ###########
    #########################################
    # Make sure the C++ compile script is executable
    run_subprocess_command([CHMOD_EXECUTABLE, CHMOD_ARGS, CPP_COMPILE_SCRIPT])
    # Compile the C++ code & run it
    run_subprocess_command([CPP_COMPILE_SCRIPT, CPP_EXECUTABLE_NAME])

##################################################################################
# MAIN PROGRAM ###################################################################
##################################################################################
"""
This script orchestrates the process of exporting an ONNX model,
embedding it into a C header file, and then compiling and running a C++
program that uses this embedded model.
"""
if __name__ == "__main__":
    try:
        main_execution_flow()
    except subprocess.CalledProcessError as e:
        print(f"Error during subprocess execution: {e}", file=sys.stderr)
        sys.exit(e.returncode)
    except FileNotFoundError as e:
        print(f"Error: A required file or command was not found: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    #########################################
    # OUTPUT RESULTS ########################
    #########################################
    # The results are printed by the C++ executable.
    # This script primarily manages the build and execution process.
    print("Script execution completed successfully.")