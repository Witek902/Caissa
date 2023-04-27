#pragma once

#include "TrainableNode.hpp"

namespace nn {

// Input node where input value can be either 0.0 or 1.0
// and most of inputs are assumed to be zero
class SparseBinaryInputNode : public ITrainableNode
{
public:
    using IndexType = uint16_t;

    class Context : public INodeContext
    {
    public:
        Context(uint32_t numOutputs) : INodeContext(numOutputs) {}
        std::span<const IndexType> sparseInputs;
    };

    SparseBinaryInputNode(uint32_t inputSize, uint32_t outputSize, const nn::WeightsStoragePtr& weights);

    virtual INodeContext* CreateContext() override
    {
        return new Context(GetNumOutputs());
    }

    void Run(INodeContext& ctx) const override;
    void Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const override;
    virtual InputMode GetInputMode() const override { return InputMode::SparseBinary; }
};

} // namespace nn
