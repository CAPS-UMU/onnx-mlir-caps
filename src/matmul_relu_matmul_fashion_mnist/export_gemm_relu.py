##############################################
# IMPORT LIBRARIES ###########################
##############################################

"""
Libraries and packages used in this script and Versions Info for tools, libraries, and packages used in this script.
"""
import os
import subprocess                # standard library, part of Python since 2.7
import onnx                     # Version: 1.16.1 (latest as of cutoff)
from onnx import helper, TensorProto
import numpy as np              # Version: 1.26.4 (latest as of cutoff)

###############################################
# CONSTANTS & PARAMETERS ######################
###############################################

"""
Constants and parameters used in this script.
"""
OP_TYPE        = 'Gemm'
OP_DOMAIN      = ''            # Default ONNX domain
ALPHA          = 1.0
BETA           = 1.0
TRANS_A        = 0             # No transpose for A
TRANS_B        = 1             # Transpose B as obtained
OPSET_CORE     = 13
PRODUCER_NAME  = 'gemm-example-producer'
MODEL_FILENAME = 'gemm_relu.onnx'
INC_FILENAME   = 'gemm_relu_model.inc'   # Name of the .inc file to generate
IR_VERSION     = 10           # Intermediate Representation version

DIMS_A = [1, 784]     # Dimensions for tensor A (input)
DIMS_B = [128, 784]   # Dimensions for tensor B (input)
DIMS_C = [128]        # Bias vector (rank-1)
DIMS_Y = [1, 128]     # Gemm output (and Relu output)

##############################################
# FUNCTION DEFINITIONS #######################
##############################################
# (none for this simple exporter)

##################################################################################
# MAIN PROGRAM ###################################################################
##################################################################################
"""
This script defines and exports a simple ONNX model containing a Gemm operator followed by a Relu operator.
Note: The model now correctly defines the final output tensor "Z" with dimensions DIMS_Y.
"""
if __name__ == "__main__":
    #########################################
    # DEFINE TENSOR VALUE INFO ##############
    #########################################
    # Define input tensors for the Gemm operation
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, DIMS_A)
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, DIMS_B)
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, DIMS_C)  # Bias tensor
    
    # Define intermediate tensor for Gemm *correctly* with DIMS_Y
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, DIMS_Y)
    # Define final output tensor for Relu
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, DIMS_Y)

    #########################################
    # CREATE GEMM NODE ######################
    #########################################
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
    # CREATE RELU NODE ######################
    #########################################
    relu_node = helper.make_node(
        "Relu",
        inputs=['Y'],
        outputs=['Z']
    )

    #########################################
    # CREATE GRAPH ##########################
    #########################################
    graph_def = helper.make_graph(
        nodes=[gemm_node, relu_node],
        name=f"{OP_TYPE}-graph",
        inputs=[A, B, C],      # C is now rank-1 [128]
        outputs=[Z]            # only Z
    )

    #########################################
    # CREATE MODEL ##########################
    #########################################
    model_def = helper.make_model(
        graph_def,
        producer_name=PRODUCER_NAME,
        opset_imports=[
            helper.make_operatorsetid(OP_DOMAIN, OPSET_CORE),  # Standard opset
            # helper.make_operatorsetid('custom.domain', OPSET_CUSTOM) # Example if using custom ops
        ]
    )
    model_def.ir_version = IR_VERSION

    #########################################
    # SAVE MODEL ############################
    #########################################
    onnx.save(model_def, MODEL_FILENAME)
    print(f"ONNX model saved to '{MODEL_FILENAME}'")

    ###########################################
    # EXPORT to .inc FILE #####################
    ###########################################
    # Uses `xxd -i` to embed the ONNX into a C header.
    with open(INC_FILENAME, 'w') as inc_f:
        subprocess.run(
            ['xxd', '-i', MODEL_FILENAME],
            stdout=inc_f,
            check=True
        )
    #print(f"C include file generated: '{INC_FILENAME}'")

    #########################################
    # OUTPUT RESULTS ########################
    #########################################
    print(f"C include file generated: '{INC_FILENAME}'")
    print(f"ONNX model '{MODEL_FILENAME}' has been successfully created and saved.")
    print(f"Model IR version: {model_def.ir_version}")
    print(f"Model producer name: {model_def.producer_name}")
    print(f"Opset import: {model_def.opset_import[0].domain} v{model_def.opset_import[0].version}")