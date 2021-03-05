#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>

class MyDialect;

class BooleanType : public mlir::Type::TypeBase<BooleanType, mlir::Type, mlir::TypeStorage> {
public:
    using Base::Base;
    static BooleanType get(mlir::MLIRContext* context) {
        return Base::get(context);
    }
};

void printMyDialectType(MyDialect* dialect, mlir::Type type, mlir::DialectAsmPrinter& printer) {
    auto& os = printer.getStream();

    if (auto tp = type.dyn_cast<BooleanType>()) {
        os << "bool";
        return;
    }
}
