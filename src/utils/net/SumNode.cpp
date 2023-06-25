#include "SumNode.hpp"

namespace nn {

SumNode::SumNode(const NodePtr& previousNodeA, const NodePtr& previousNodeB)
    : ICombiningNode(previousNodeA->GetNumOutputs(), previousNodeA->GetNumOutputs())
{
    ASSERT(previousNodeA->GetNumOutputs() == previousNodeB->GetNumOutputs());

    m_inputNodes[0] = previousNodeA.get();
    m_inputNodes[1] = previousNodeB.get();
}

void SumNode::Run(INodeContext& ctx) const
{
    Context& context = static_cast<Context&>(ctx);

    ASSERT(context.outputs.size() == m_numOutputs);
    ASSERT(context.inputs.size() == m_inputNodes[0]->GetNumOutputs());
    ASSERT(context.secondaryInputs.size() == m_inputNodes[1]->GetNumOutputs());

    for (uint32_t i = 0; i < context.inputs.size(); ++i)
    {
        context.outputs[i] = context.inputs[i] + context.secondaryInputs[i];
    }
}

void SumNode::Backpropagate(const Values& error, INodeContext& ctx, Gradients&) const
{
    Context& context = static_cast<Context&>(ctx);

    ASSERT(context.outputs.size() == error.size());
    ASSERT(context.outputs.size() == GetNumOutputs());
    ASSERT(context.inputError.size() == m_inputNodes[0]->GetNumOutputs());
    ASSERT(context.secondaryInputError.size() == m_inputNodes[1]->GetNumOutputs());

    for (uint32_t i = 0; i < context.outputs.size(); ++i)
    {
        context.inputError[i] = error[i];
        context.secondaryInputError[i] = error[i];
    }
}

} // namespace nn
