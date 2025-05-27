##############################################
# IMPORT LIBRARIES ###########################
##############################################
"""
Libraries and packages used in this script and Versions Info for tools, libraries, and packages used in this script.
"""
import onnx  # Version: 1.16.1 (latest as of cutoff)
from onnx import helper, TensorProto # Part of ONNX library
import numpy as np # Version: 1.26.4 (latest as of cutoff)

###############################################
# CONSTANTS & PARAMETERS ######################
###############################################
"""
Constants and parameters used in this script.
"""
# ONNX Model Parameters
OP_TYPE = 'Gemm'
OP_DOMAIN = ''  # Default ONNX domain can be indicated as empty string
ALPHA = 1.0
BETA = 1.0
TRANS_A = 0  # No transpose for A
TRANS_B = 0  # No transpose for B
OPSET_CORE = 13
OPSET_CUSTOM = 1 # Custom opset version (though not used for standard Gemm)
PRODUCER_NAME = 'gemm-example-producer'
MODEL_FILENAME = 'gemm.onnx'
IR_VERSION = 10 # Intermediate Representation version

# Tensor Dimensions
DIMS_A = [2, 3] # Dimensions for tensor A (input)
DIMS_B = [3, 2] # Dimensions for tensor B (input)
DIMS_C = [2, 2] # Dimensions for tensor C (input, bias) and Y (output)

##############################################
# FUNCTION DEFINITIONS #######################
##############################################
"""
Functions must adhere to the following structure:

1) Use docstrings to describe the function's purpose, parameters, and return values.
2) Be named according to their functionality.
"""
# No functions defined for this simple export script.
# ONNX helper functions are used directly.

##################################################################################
# MAIN PROGRAM ###################################################################
##################################################################################
"""
This script defines and exports a simple ONNX model containing a single Gemm operator.
"""
if __name__ == "__main__":
    #########################################
    # DEFINE TENSOR VALUE INFO ##############
    #########################################
    # Define input and output tensors for the Gemm operation
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, DIMS_A)
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, DIMS_B)
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, DIMS_C) # Bias tensor
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, DIMS_C) # Output tensor

    #########################################
    # CREATE GEMM NODE ######################
    #########################################
    # Create the Gemm node with specified inputs, outputs, and attributes
    gemm_node = helper.make_node(
        OP_TYPE,
        inputs=['A', 'B', 'C'],
        outputs=['Y'],
        alpha=ALPHA,
        beta=BETA,
        transA=TRANS_A,
        transB=TRANS_B,
        domain=OP_DOMAIN
    )

    #########################################
    # CREATE GRAPH ##########################
    #########################################
    # Create the graph containing the Gemm node
    graph_def = helper.make_graph(
        nodes=[gemm_node],
        name=f"{OP_TYPE}-graph",
        inputs=[A, B, C],
        outputs=[Y]
    )

    #########################################
    # CREATE MODEL ##########################
    #########################################
    # Create the ONNX model
    model_def = helper.make_model(
        graph_def,
        producer_name=PRODUCER_NAME,
        opset_imports=[
            helper.make_operatorsetid(OP_DOMAIN, OPSET_CORE), # Standard opset
            # helper.make_operatorsetid('custom.domain', OPSET_CUSTOM) # Example if using custom ops
        ]
    )
    model_def.ir_version = IR_VERSION

    #########################################
    # SAVE MODEL ############################
    #########################################
    # Save the ONNX model to a file
    onnx.save(model_def, MODEL_FILENAME)

    #########################################
    # OUTPUT RESULTS ########################
    #########################################
    print(f"ONNX model '{MODEL_FILENAME}' has been successfully created and saved.")
    print(f"Model IR version: {model_def.ir_version}")
    print(f"Model producer name: {model_def.producer_name}")
    print(f"Opset import: {model_def.opset_import[0].domain} v{model_def.opset_import[0].version}")