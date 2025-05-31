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
import argparse, json

###############################################
# CONSTANTS & PARAMETERS ######################
###############################################

"""
Constants and parameters used in this script.
"""
tparser = argparse.ArgumentParser(description="Export Gemm+ReLU ONNX model based on JSON config.")
tparser.add_argument('--config', type=str, default='config.json', help='Path to JSON config file')
config_args = tparser.parse_args()
with open(config_args.config, 'r') as cf:
    cfg = json.load(cf)

# Load parameters from config
OP_TYPE = cfg['export_gemm']['op_type']
OP_DOMAIN = cfg['export_gemm']['op_domain']
ALPHA = cfg['export_gemm']['alpha']
BETA = cfg['export_gemm']['beta']
TRANS_A = cfg['export_gemm']['transA']
TRANS_B = cfg['export_gemm']['transB']
OPSET_CORE = cfg['export_gemm']['opset']
PRODUCER_NAME = cfg['export_gemm']['producer_name']
MODEL_FILENAME = cfg['export_gemm']['model_filename']
INC_FILENAME = cfg['export_gemm']['inc_filename']
IR_VERSION = cfg['export_gemm']['ir_version']
DIMS_A = cfg['export_gemm']['dims_A']
DIMS_B = cfg['export_gemm']['dims_B']
DIMS_C = cfg['export_gemm']['dims_C']
DIMS_Y = cfg['export_gemm']['dims_Y']

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