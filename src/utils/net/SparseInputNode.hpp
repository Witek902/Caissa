#pragma once

#include "TrainableNode.hpp"

namespace nn {

// Input node where  most of inputs are assumed to be zero
class SparseInputNode : public ITrainableNode
{
public:
    class Context : public INodeContext
    {
    public:
        Context(uint32_t numOutputs) : INodeContext(numOutputs) {}
        std::span<const ActiveFeature> sparseInputs;
    };

    SparseInputNode(uint32_t inputSize, uint32_t outputSize);

    virtual INodeContext* CreateContext() override
    {
        return new Context(GetNumOutputs());
    }

    virtual void Run(INodeContext& ctx) const override;
    virtual void Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const override;
    virtual InputMode GetInputMode() const override { return InputMode::Sparse; }
    virtual bool IsInputNode() const override { return true; }
};

} // namespace nn
