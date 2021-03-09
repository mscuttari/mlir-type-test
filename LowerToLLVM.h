#pragma once

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/LLVMIR/FunctionCallUtils.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/Transforms.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinOps.h>

#include "Ops.h"
#include "TypeConverter.h"

class TestOpLowering : public mlir::OpConversionPattern<TestOp>
{
public:
    TestOpLowering(mlir::MLIRContext* ctx, mlir::TypeConverter& typeConverter)
            : mlir::OpConversionPattern<TestOp>(typeConverter, ctx, 1)
    {
    }

    mlir::LogicalResult matchAndRewrite(TestOp op, llvm::ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter) const override
    {
        mlir::Location location = op.getLoc();

        // Split the current block
        mlir::Block* currentBlock = rewriter.getInsertionBlock();
        mlir::Block* continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        // Inline regions
        mlir::Block* conditionBlock = &op.condition().front();
        mlir::Block* bodyBlock = &op.body().front();

        rewriter.inlineRegionBefore(op.body(), continuation);
        rewriter.inlineRegionBefore(op.condition(), bodyBlock);

        // Start the loop by branching to the "condition" region
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<mlir::BranchOp>(location, conditionBlock, op.getOperands());

        // Replace the "condition" block terminator with a conditional branch
        rewriter.setInsertionPointToEnd(conditionBlock);
        auto conditionOp = mlir::cast<ConditionOp>(conditionBlock->getTerminator());
        rewriter.replaceOpWithNewOp<mlir::CondBranchOp>(conditionOp, conditionOp->getOperand(0), bodyBlock, conditionOp.args(), continuation, llvm::None);

        // Replace "body" block terminator with a branch to the "step" block
        rewriter.setInsertionPointToEnd(bodyBlock);
        auto bodyYieldOp = mlir::cast<YieldOp>(bodyBlock->getTerminator());
        rewriter.replaceOpWithNewOp<mlir::BranchOp>(bodyYieldOp, conditionBlock, bodyYieldOp.getOperands());

        rewriter.eraseOp(op);
        return mlir::success();
    }
};

void populateMyDialectToLLVMConversionPatterns(mlir::OwningRewritePatternList& patterns, mlir::MLIRContext* context, TypeConverter& typeConverter) {
    patterns.insert<TestOpLowering>(context, typeConverter);
}

class MyDialectToLLVMLoweringPass : public mlir::PassWrapper<MyDialectToLLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    MyDialectToLLVMLoweringPass() {

    }

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::StandardOpsDialect>();
        registry.insert<mlir::LLVM::LLVMDialect>();
    }

    mlir::LogicalResult step1(mlir::ModuleOp module) {
      mlir::ConversionTarget target(getContext());
      target.addLegalDialect<mlir::StandardOpsDialect, mlir::BuiltinDialect>();

      mlir::TypeConverter typeConverter;

      typeConverter.addConversion([](mlir::Type type) ->llvm::Optional<mlir::Type> {
        if (llvm::isa<mlir::BuiltinDialect>(type.getDialect()))
          return type;
        return llvm::None;
      });

      typeConverter.addConversion([this](BooleanType type) {
        return mlir::IntegerType::get(&getContext(), 1);
      });

      typeConverter.addTargetMaterialization([](mlir::OpBuilder &builder,
                                                mlir::IntegerType resultType,
                                                mlir::ValueRange inputs,
                                                mlir::Location loc)
                                             -> llvm::Optional<mlir::Value> {
        if (inputs.size() != 1 || !inputs[0].getType().isa<BooleanType>())
          return llvm::None;

        return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
      });

      typeConverter.addSourceMaterialization([](mlir::OpBuilder &builder,
                                                BooleanType resultType,
                                                mlir::ValueRange inputs,
                                                mlir::Location loc)
                                             -> llvm::Optional<mlir::Value> {
        if (inputs.size() != 1 || !inputs[0].getType().isa<mlir::IntegerType>())
          return llvm::None;

        return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
      });

      mlir::OwningRewritePatternList patterns;
      patterns.insert<TestOpLowering>(&getContext(), typeConverter);
      return applyPartialConversion(module, target, std::move(patterns));
    }

    void runOnOperation() final {
        auto module = getOperation();

        if (failed(step1(module)))
          return signalPassFailure();
#if 0
        mlir::ConversionTarget target(getContext());
        target.addLegalDialect<mlir::LLVM::LLVMDialect>();
        target.addIllegalOp<mlir::LLVM::DialectCastOp>();
        target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp, mlir::UnrealizedConversionCastOp>();

        // We need to mark the scf::YieldOp and AffineYieldOp as legal due to a
        // current limitation of MLIR. In fact, they are used just as a placeholder
        // and would lead to conversion problems if encountered while lowering.
        target.addLegalOp<mlir::scf::YieldOp>();

        // Create the Modelica types converter. We also need to create
        // the functions wrapper, in order to JIT it easily.
        mlir::LowerToLLVMOptions llvmLoweringOptions;
        llvmLoweringOptions.emitCWrappers = true;
        TypeConverter typeConverter(&getContext(), llvmLoweringOptions);

        // Provide the set of patterns that will lower the Modelica operations
        mlir::OwningRewritePatternList patterns;
        populateStdToLLVMConversionPatterns(typeConverter, patterns);
        populateLoopToStdConversionPatterns(patterns, &getContext());
        //mlir::scf::populateSCFStructuralTypeConversionsAndLegality(&getContext(), typeConverter, patterns, target);
        populateMyDialectToLLVMConversionPatterns(patterns, &getContext(), typeConverter);

        // With the target and rewrite patterns defined, we can now attempt the
        // conversion. The conversion will signal failure if any of our "illegal"
        // operations were not converted successfully.
        if (failed(applyPartialConversion(module, target, std::move(patterns))))
        {
        mlir::emitError(module.getLoc(), "Error in converting to LLVM dialect\n");
        signalPassFailure();
        }
#endif
    }
};

std::unique_ptr<mlir::Pass> createMyDialectToLLVMLoweringPass() {
    return std::make_unique<MyDialectToLLVMLoweringPass>();
}
