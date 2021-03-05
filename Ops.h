#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

//===----------------------------------------------------------------------===//
// TestOp
//===----------------------------------------------------------------------===//

class TestOp : public mlir::Op<TestOp, mlir::OpTrait::NRegions<2>::Impl, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult>
{
public:
    using Op::Op;

    static llvm::StringRef getOperationName() {
        return "mydialect.test";
    }

    mlir::Region& condition() {
        return getRegion(0);
    }

    mlir::Region& body() {
        return getRegion(1);
    }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange args) {
        state.addOperands(args);
        auto insertionPoint = builder.saveInsertionPoint();

        builder.createBlock(state.addRegion(), {}, args.getTypes());
        builder.createBlock(state.addRegion(), {}, args.getTypes());

        builder.restoreInsertionPoint(insertionPoint);
    }
};

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

class ConditionOp : public mlir::Op<ConditionOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::ZeroSuccessor, mlir::OpTrait::IsTerminator> {
public:
    using Op::Op;

    static llvm::StringRef getOperationName() {
        return "mydialect.condition";
    }

    static void build(mlir::OpBuilder& builder, ::mlir::OperationState& state, mlir::Value condition, mlir::ValueRange args = {}) {
        state.addOperands(condition);
        state.addOperands(args);
    }

    mlir::Value condition() {
        return getOperand(0);
    }

    mlir::ValueRange args() {
        mlir::OperandRange range(std::next(getOperands().begin()), getOperands().end());
        return mlir::ValueRange(range);
    }
};

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

class YieldOp : public mlir::Op<YieldOp, mlir::OpTrait::ZeroRegion, mlir::OpTrait::VariadicOperands, mlir::OpTrait::ZeroResult, mlir::OpTrait::IsTerminator>
{
public:
    using Op::Op;

    static llvm::StringRef getOperationName() {
        return "mydialect.yield";
    }

    static void build(mlir::OpBuilder& builder, mlir::OperationState& state, mlir::ValueRange args = {}) {
        state.addOperands(args);
    }

    mlir::ValueRange args() {
        return getOperands();
    }
};