#include <iostream>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/StandardOps/Transforms/Passes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/IR/MLIRContext.h>

#include "LowerToLLVM.h"
#include "MyDialect.h"
#include "Ops.h"
#include "TypeConverter.h"
#include "Types.h"

int main() {
    mlir::MLIRContext context;
    context.loadDialect<MyDialect>();
    context.loadDialect<mlir::StandardOpsDialect>();

    mlir::OpBuilder builder(&context);
    mlir::Location loc = builder.getUnknownLoc();

    // Create module and function
    mlir::ModuleOp module = mlir::ModuleOp::create(loc);
    auto functionType = builder.getFunctionType(llvm::None, llvm::None);
    auto function = mlir::FuncOp::create(loc, "main", functionType);
    module.push_back(function);

    // Create function body
    auto& entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    {
        /// Create test operation
        mlir::Value start = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(0));
        auto testOp = builder.create<TestOp>(loc, start);

        // Condition
        builder.setInsertionPointToStart(&testOp.condition().front());
        builder.create<ConditionOp>(
                loc,
                builder.create<mlir::ConstantOp>(loc, builder.getBoolAttr(true)),
                testOp.condition().front().getArgument(0));

        // Body
        builder.setInsertionPointToStart(&testOp.body().front());
        mlir::Value oneValue = builder.create<mlir::ConstantOp>(loc, builder.getIndexAttr(1));
        mlir::Value nextValue = builder.create<mlir::AddIOp>(loc, testOp.body().front().getArgument(0), oneValue);
        builder.create<YieldOp>(loc, nextValue);

        builder.setInsertionPointAfter(testOp);
    }

    builder.create<mlir::ReturnOp>(loc);

    llvm::DebugFlag = true;

    // Dump the module
    module.dump();

    // Convert the module to LLVM IR
    mlir::PassManager passManager(&context);
    passManager.addPass(createMyDialectToLLVMLoweringPass());
    passManager.run(module);

    return 0;
}
