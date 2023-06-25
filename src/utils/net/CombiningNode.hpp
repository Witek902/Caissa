#pragma once

#include "Node.hpp"

namespace nn {

// combine outputs of two nodes
class ICombiningNode : public INode
{
public:
    class Context : public INodeContext
    {
    public:
        Context(const ICombiningNode& node)
            : INodeContext(node.GetNumOutputs())
        {
            inputError.resize(node.m_inputNodes[0]->GetNumOutputs());
            secondaryInputError.resize(node.m_inputNodes[1]->GetNumOutputs());
        }

        std::span<const float> secondaryInputs; // saved by INode::Run()
        Values secondaryInputError;             // saved by INode::Backpropagate()
    };

    ICombiningNode(uint32_t numInputs, uint32_t numOutputs)
        : INode(numInputs, numOutputs)
        , m_inputNodes{}
    {}

    virtual INodeContext* CreateContext() override
    {
        return new Context(*this);
    }

    virtual InputMode GetInputMode() const override { return InputMode::Full; }
    virtual bool IsCombining() const override { return true; }

    INLINE const INode* GetInputNode(uint32_t index) const { return m_inputNodes[index]; }

protected:
    const INode* m_inputNodes[2];
};

} // namespace nn
