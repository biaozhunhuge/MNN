//
//  MatMulOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(MatrixMultiplicationOnnx);

MNN::OpType MatrixMultiplicationOnnx::opType() {
    return MNN::OpType_MatrixMultiplication;
}
MNN::OpParameter MatrixMultiplicationOnnx::type() {
    return MNN::OpParameter_NONE;
}

void MatrixMultiplicationOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                     std::vector<const onnx::TensorProto*> initializers) {
}

REGISTER_CONVERTER(MatrixMultiplicationOnnx, MatMul);
