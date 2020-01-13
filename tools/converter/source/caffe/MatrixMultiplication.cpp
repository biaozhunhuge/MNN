//
//  MatrixMultiplication.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "OpConverter.hpp"

class MatrixMultiplication : public OpConverter {
public:
    virtual void run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight);
	MatrixMultiplication() {
    }
    virtual ~MatrixMultiplication() {
    }
    virtual MNN::OpType opType() {
        return MNN::OpType_MatrixMultiplication;
    }
    virtual MNN::OpParameter type() {
        return MNN::OpParameter_NONE;
    }
};

void MatrixMultiplication::run(MNN::OpT* dstOp, const caffe::LayerParameter& parameters, const caffe::LayerParameter& weight) {

}

static OpConverterRegister<MatrixMultiplication> b("MatrixMultiplication");
