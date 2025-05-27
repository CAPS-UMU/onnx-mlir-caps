#!/usr/bin/env bash

##############################################
# SCRIPT CONFIGURATION #######################
##############################################
#
# Script configuration and initial setup.
#
set -e # Exit immediately if a command exits with a non-zero status.

###############################################
# CONSTANTS & PARAMETERS ######################
###############################################
#
# Constants and parameters used in this script.
#
ORT_INSTALL_DIR="onnxruntime-linux-x64-1.22.0" # ONNX Runtime installation directory. Version: 1.22.0 (Assumed based on path)
ORT_INC="${ORT_INSTALL_DIR}/include"
ORT_LIB="${ORT_INSTALL_DIR}/lib"

# Script arguments
EXECUTABLE_NAME="" # Will be set based on the first script argument

##############################################
# FUNCTION DEFINITIONS #######################
##############################################
#
# Functions must adhere to the following structure:
#
# 1) Use comments to describe the function's purpose, parameters, and return values.
# 2) Be named according to their functionality.
#

# No functions defined in this script.

##################################################################################
# MAIN PROGRAM ###################################################################
##################################################################################
#
# Main execution flow of the script.
#

#########################################
# VALIDATE ARGUMENTS ####################
#########################################
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <test_name (without .cpp)>"
  exit 1
fi

EXECUTABLE_NAME="$1"
SOURCE_FILE="${EXECUTABLE_NAME}.cpp"
BINARY_FILE="${EXECUTABLE_NAME}"

if [ ! -f "$SOURCE_FILE" ]; then
  echo "Error: source file '$SOURCE_FILE' not found."
  exit 1
fi

#########################################
# COMPILE SOURCE CODE ###################
#########################################
echo "Compiling $SOURCE_FILE → $BINARY_FILE"
g++ -std=c++17 -O2 \
  -I"${ORT_INC}" \
  "$SOURCE_FILE" \
  -o "$BINARY_FILE" \
  -L"${ORT_LIB}" -lonnxruntime -pthread

#########################################
# EXECUTE COMPILED PROGRAM ##############
#########################################
echo "Running ./${BINARY_FILE}"
export LD_LIBRARY_PATH="${ORT_LIB}:${LD_LIBRARY_PATH}"
./"$BINARY_FILE"

#########################################
# OUTPUT RESULTS ########################
#########################################
# The C++ executable prints its own results.
# This script primarily manages the build and execution process.
echo "Script execution completed successfully for ${BINARY_FILE}."