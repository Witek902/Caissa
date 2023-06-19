#pragma once

#include "Node.hpp"

namespace nn {

// concatenate outputs of two nodes
class ConcatenationNode : public INode
{
public:
    class Context : public INodeContext
    {
    public:
        Context(const ConcatenationNode& node)
            : INodeContext(node.GetNumOutputs())
        {
            inputError.resize(node.m_inputNodes[0]->GetNumOutputs());
            secondaryInputError.resize(node.m_inputNodes[1]->GetNumOutputs());
        }

        std::span<const float> secondaryInputs; // saved by INode::Run()
        Values secondaryInputError;             // saved by INode::Backpropagate()
    };

    ConcatenationNode(const NodePtr& previousNodeA, const NodePtr& previousNodeB);

    virtual INodeContext* CreateContext() override
    {
        return new Context(*this);
    }

    virtual void Run(INodeContext& ctx) const override;
    virtual void Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const override;
    virtual InputMode GetInputMode() const override { return InputMode::Full; }
    virtual bool IsConcatenation() const override { return true; }

    INLINE const INode* GetInputNode(uint32_t index) const { return m_inputNodes[index]; }

private:
    const INode* m_inputNodes[2];
};

} // namespace nn
