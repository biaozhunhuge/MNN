//
//  MergeOptimizer.hpp
//  MNN
//
//  Created by MNN on 2019/08/20.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#ifndef MergeOptimizer_hpp
#define MergeOptimizer_hpp

#include "Optimizer.hpp"
#include "MNNForwardType.h"
namespace MNN {
namespace Express {
class MergeOptimizer : public Optimizer {
public:
    virtual ~MergeOptimizer() = default;
    MergeOptimizer(MNNForwardType type, int numberThread, BackendConfig* config);

    virtual Cost onMeasure(const Model& model, std::shared_ptr<Parameters> parameters) override;
    virtual bool onExecute(Model& model, std::shared_ptr<Parameters> parameters) override;

private:
    BackendConfig mConfig;
    MNNForwardType mType;
    int mNumberThread;
};
}; // namespace Express
}; // namespace MNN
#endif
