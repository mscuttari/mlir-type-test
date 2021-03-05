#pragma once

#include <mlir/IR/Dialect.h>

#include "Ops.h"
#include "Types.h"

class MyDialect : public mlir::Dialect
{
public:
    explicit MyDialect(mlir::MLIRContext* context) : mlir::Dialect("mydalect", context, mlir::TypeID::get<MyDialect>()){
        addTypes<BooleanType>();
        addOperations<TestOp, ConditionOp, YieldOp>();
    }

    static llvm::StringRef getDialectNamespace() {
        return "mydialect";
    }
};
