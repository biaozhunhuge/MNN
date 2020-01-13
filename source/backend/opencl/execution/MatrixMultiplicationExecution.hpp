//
//  InterpExecution.hpp
//  MNN
//
//  Created by MNN on 2019/02/02.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef MatrixMultiplicationExecution_hpp
#define MatrixMultiplicationExecution_hpp

#include <array>
#include <memory>
#include <vector>
#include "Execution.hpp"
#include "core/OpenCLBackend.hpp"

namespace MNN {
namespace OpenCL {

class MatrixMultiplicationExecution : public Execution {
public:
	MatrixMultiplicationExecution(const std::vector<Tensor *> &inputs, const MNN::Op *op, Backend *backend);
    virtual ~MatrixMultiplicationExecution() = default;

	virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    std::vector<uint32_t> interpLocalWS(const std::vector<uint32_t> &gws, const uint32_t maxWorkGroupSize);

private:
    cl::Kernel mKernel;
	cl::Kernel mBufferToImageKernel;
	cl::Kernel mImageToBufferKernel;

	uint32_t mMaxWGS_M;
    bool mAreadySetArg;
    OpenCLBackend *mOpenCLBackend;

	std::shared_ptr<Tensor> mInputTemp;
	std::shared_ptr<Tensor> mBTemp;
	std::shared_ptr<Tensor> mOutputTemp;
};

} // namespace OpenCL
} // namespace MNN
#endif /* MatrixMultiplicationExecution_hpp */
