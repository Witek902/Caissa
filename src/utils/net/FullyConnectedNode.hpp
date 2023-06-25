#pragma once

#include "TrainableNode.hpp"

namespace nn {

// regular fully connected dense layer
class FullyConnectedNode : public ITrainableNode
{
public:
    class Context : public INodeContext
    {
    public:
        Context(const FullyConnectedNode& node)
            : INodeContext(node.GetNumOutputs())
        {
            inputError.resize(node.GetNumInputs());
        }
    };

    FullyConnectedNode(const NodePtr& previousNode, uint32_t inputSize, uint32_t outputSize, const nn::WeightsStoragePtr& weights);

    virtual INodeContext* CreateContext() override
    {
        return new Context(*this);
    }

    virtual void Run(INodeContext& ctx) const override;
    virtual void Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const override;
    virtual InputMode GetInputMode() const override { return InputMode::Full; }
    virtual bool IsInputNode() const override { return m_previousNode == nullptr; }
};


} // namespace nn
