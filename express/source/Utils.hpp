//
//  Utils.hpp
//  MNN
//
//  Created by MNN on 2019/07/26.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "Expr.hpp"
#include "Tensor.hpp"
namespace MNN {
namespace Express {
class Utils {
public:
    static void copyInfoToTensor(Tensor* dest, const Variable::Info* source);
    static void copyTensorToInfo(Variable::Info* dest, const Tensor* source);
};
} // namespace Express
} // namespace MNN
