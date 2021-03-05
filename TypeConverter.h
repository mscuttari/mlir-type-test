#pragma once

#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/MLIRContext.h>
#include "Types.h"

class TypeConverter : public mlir::LLVMTypeConverter {
public:
    TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options) : mlir::LLVMTypeConverter(context, options)
    {
        addConversion([&](BooleanType type) {
            return convertType(mlir::IntegerType::get(&getContext(), 1));
        });

        addTargetMaterialization(
                [&](mlir::OpBuilder &builder, mlir::IntegerType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
                    if (inputs.size() != 1)
                        return llvm::None;

                    if (!inputs[0].getType().isa<BooleanType>())
                        return llvm::None;

                    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
                });

        addSourceMaterialization(
                [&](mlir::OpBuilder &builder, BooleanType resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
                    if (inputs.size() != 1)
                        return llvm::None;

                    if (!inputs[0].getType().isa<mlir::IntegerType>() || inputs[0].getType().getIntOrFloatBitWidth() != 1)
                        return llvm::None;

                    return builder.create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs[0]).getResult(0);
                });

    }
};