//
//  CPUBatchMatMul.cpp
//  MNN
//
//  Created by MNN on 2019/03/25.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CPUMatrixMultiplication.hpp"
#include "CPUBackend.hpp"
#include "Matrix.hpp"

namespace MNN {

	CPUMatrixMultiplication::CPUMatrixMultiplication(const Op* op, Backend* backend) : Execution(backend) {
		// nothing to do
	}

	ErrorCode CPUMatrixMultiplication::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
		auto input0 = inputs[0];
		auto input1 = inputs[1];
		auto output = outputs[0];
		const int dimensions = input0->dimensions();

		int batch = 1;
		for (int i = 0; i < dimensions - 2; ++i) {
			batch *= input0->length(i);
		}
		mBatch = batch;

		std::vector<int> dimSizes(2);

		dimSizes[0] = input0->length(dimensions - 2);
		dimSizes[1] = input0->length(dimensions - 1);
		mMatrixA.reset(Tensor::createDevice<float>(dimSizes));

		dimSizes[0] = input1->length(dimensions - 2);
		dimSizes[1] = input1->length(dimensions - 1);
		mMatrixB.reset(Tensor::createDevice<float>(dimSizes));

		dimSizes[0] = output->length(dimensions - 2);
		dimSizes[1] = output->length(dimensions - 1);
		mMatrixC.reset(Tensor::createDevice<float>(dimSizes));

		return NO_ERROR;
	}

	void align_memory(float* src, int h, int w){
		float* dst = new float[h * w];
		memcpy(dst, src, h * w * sizeof(float));
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				int num = (int(i / 4) * 3 + j);
				int idx = num * 4 + (i % 4);
				dst[i*3+j] = src[idx];
			}
		}

		memcpy(src, dst, h * w * sizeof(float));
		delete[] dst;
		int a = 0;

	}

	void align_memory_inverse(float* src, int h, int w) {
		int nh = (int(h / 4) +1) * 4;
		float* dst = new float[nh * w];
		memset(dst, 0, nh * w * sizeof(float));

		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				int num = (int(i / 4) * 3 + j);
				int idx = num * 4 + (i % 4);
				dst[idx] = src[i * 3 + j];
			}
		}

		memcpy(src, dst, nh * w * sizeof(float));
		delete[] dst;
		int a = 0;

	}

	ErrorCode CPUMatrixMultiplication::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
		auto input0 = inputs[0];
		auto input1 = inputs[1];
		auto output = outputs[0];
		const int dimensions = input0->dimensions();
		auto in_n = input0->length(0);
		auto in_c = input0->length(1);
		auto in_h = input0->length(2);
		auto in_w = input0->length(3);

		MNN_ASSERT(dimensions >= 3);
		const int input0Stride = input0->stride(dimensions - 3);
		const int input1Stride = input1->stride(dimensions - 3);
		const int outputStride = output->stride(dimensions - 3);
		const auto input0Ptr = input0->host<float>();
		const auto input1Ptr = input1->host<float>();
		float* const outputPtr = output->host<float>();

		for (int i = 0; i < mBatch; ++i) {
			mMatrixA->buffer().host = reinterpret_cast<uint8_t*>(input0Ptr + i * input0Stride);
			mMatrixB->buffer().host = reinterpret_cast<uint8_t*>(input1Ptr + i * input1Stride);
			mMatrixC->buffer().host = reinterpret_cast<uint8_t*>(outputPtr + i * outputStride);

			Math::Matrix::gv_multi(mMatrixC.get(), mMatrixA.get(), mMatrixB.get());
		}

		return NO_ERROR;
	}

	class CPUMatrixMultiplicationCreator : public CPUBackend::Creator {
	public:
		virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs,
			const MNN::Op* op, Backend* backend) const override {
			return new CPUMatrixMultiplication(op, backend);
		}
	};

	REGISTER_CPU_OP_CREATOR(CPUMatrixMultiplicationCreator, OpType_MatrixMultiplication);

} // namespace MNN
