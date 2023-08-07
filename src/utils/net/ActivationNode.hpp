#pragma once

#include "Node.hpp"

namespace nn {

enum class ActivationFunction : uint8_t
{
    Linear,
    ReLU,
    CReLU,      // clipped ReLU
    SqrCReLU,   // squared clipped ReLU
    Sigmoid,
    EvalToGameScore,
};

// node applying an activation function
class ActivationNode : public INode
{
public:
    class Context : public INodeContext
    {
    public:
        Context(const ActivationNode& node)
            : INodeContext(node.GetNumOutputs())
        {
            inputError.resize(node.GetNumInputs());
        }
    };

    // TODO concatenation of two nodes
    ActivationNode(const NodePtr& previousNode, ActivationFunction func = ActivationFunction::ReLU);

    virtual INodeContext* CreateContext() override
    {
        return new Context(*this);
    }

    virtual void Run(INodeContext& ctx) const override;
    virtual void Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const override;
    virtual InputMode GetInputMode() const override { return InputMode::Full; }

private:
    ActivationFunction mActivationFunc;
};

} // namespace nn
