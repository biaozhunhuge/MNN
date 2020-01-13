//
//  InterpExecution.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "execution/MatrixMultiplicationExecution.hpp"
#include "TensorUtils.hpp"

namespace MNN {
namespace OpenCL {

MatrixMultiplicationExecution::MatrixMultiplicationExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend)
    : Execution(backend) {
    mOpenCLBackend = static_cast<OpenCLBackend *>(backend);
    auto runtime   = mOpenCLBackend->getOpenCLRuntime();

    std::set<std::string> buildOptions;
    mKernel = runtime->buildKernel("MatrixMultiplication", "MatrixMultiplication", buildOptions);

	mMaxWGS_M = static_cast<uint32_t>(mOpenCLBackend->getOpenCLRuntime()->getMaxWorkGroupSize(mKernel));
    mAreadySetArg = false;
}


ErrorCode MatrixMultiplicationExecution::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
	auto bufferPool = mOpenCLBackend->getBufferPool();
	mInputTemp.reset(Tensor::createDevice<float>(tensorShapeFormat(inputs[0])));
	mBTemp.reset(Tensor::createDevice<float>(tensorShapeFormat(inputs[1])));
	mOutputTemp.reset(Tensor::createDevice<float>(tensorShapeFormat(outputs[0])));

	auto inputBuffer = bufferPool->alloc(mInputTemp->size());
	auto inputB = bufferPool->alloc(mBTemp->size());
	auto outputBuffer = bufferPool->alloc(mOutputTemp->size());

	mInputTemp->buffer().device = (uint64_t)inputBuffer;
	mBTemp->buffer().device = (uint64_t)inputB;
	mOutputTemp->buffer().device = (uint64_t)outputBuffer;

	bufferPool->recycle(inputBuffer);
	bufferPool->recycle(inputB);
	bufferPool->recycle(outputBuffer);

	return NO_ERROR;
}

ErrorCode MatrixMultiplicationExecution::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
#ifdef LOG_VERBOSE
    MNN_PRINT("Start InterpExecution onExecute... \n");
#endif
    Tensor *input  = inputs[0];
	Tensor* B = inputs[1];
    Tensor *output = outputs[0];

	std::vector<int> inputShape = tensorShapeFormat(input);
	std::vector<int> BShape = tensorShapeFormat(B);
	std::vector<int> outputShape = tensorShapeFormat(output);

	int iN = inputShape.at(0);
	int iH = inputShape.at(1);
	int iW = inputShape.at(2);
	int iC = inputShape.at(3);
	int h = iC;
	int k = iH;
	int w = B->length(2);

	convertImageToNCHWBuffer(input, mInputTemp.get(), mImageToBufferKernel, mOpenCLBackend->getOpenCLRuntime());
	convertImageToNCHWBuffer(B, mBTemp.get(), mImageToBufferKernel, mOpenCLBackend->getOpenCLRuntime());

	{
		std::vector<uint32_t> gws = { static_cast<uint32_t>(iN), static_cast<uint32_t>(iC), static_cast<uint32_t>(iH) };
		const std::vector<uint32_t> lws = { 4, 4, 4 };
		int32_t shape[4] = { h, w, k, iN };
		int32_t strides_batch[4] = { iC*iH, BShape.at(1) * BShape.at(3), outputShape.at(1)* outputShape.at(3), 0 };
		int32_t strides[4] = { input->stride(1), B->stride(1), output->stride(1),0 };

		uint32_t idx = 0;
		mKernel.setArg(idx++, openCLBuffer(mInputTemp.get()));
		mKernel.setArg(idx++, openCLBuffer(mBTemp.get()));
		mKernel.setArg(idx++, openCLBuffer(mOutputTemp.get()));
		mKernel.setArg(idx++, shape);
		mKernel.setArg(idx++, strides_batch);
		mKernel.setArg(idx++, strides);

		run3DKernelDefault(mKernel, gws, lws, mOpenCLBackend->getOpenCLRuntime());
	}
	convertNCHWBufferToImage(mOutputTemp.get(), output, mBufferToImageKernel, mOpenCLBackend->getOpenCLRuntime());

#ifdef LOG_VERBOSE
    MNN_PRINT("end InterpExecution onExecute... \n");
#endif

    return NO_ERROR;
}

class MatrixMultiplicationCreator : public OpenCLBackend::Creator {
public:
	virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
		const MNN::Op* op, Backend* backend) const override {

		return new MatrixMultiplicationExecution(inputs, op, backend);
	}
};

OpenCLCreatorRegister<MatrixMultiplicationCreator> __MatrixMultiplication_op_(OpType_MatrixMultiplication);
} // namespace OpenCL
} // namespace MNN
