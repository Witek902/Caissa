#include "ConcatenationNode.hpp"

namespace nn {

ConcatenationNode::ConcatenationNode(const NodePtr& previousNodeA, const NodePtr& previousNodeB)
    : ICombiningNode(previousNodeA->GetNumOutputs() + previousNodeB->GetNumOutputs(),
                     previousNodeA->GetNumOutputs() + previousNodeB->GetNumOutputs())
{
    m_inputNodes[0] = previousNodeA.get();
    m_inputNodes[1] = previousNodeB.get();
}

void ConcatenationNode::Run(INodeContext& ctx) const
{
    Context& context = static_cast<Context&>(ctx);

    ASSERT(context.outputs.size() == m_numOutputs);
    ASSERT(context.inputs.size() == m_inputNodes[0]->GetNumOutputs());
    ASSERT(context.secondaryInputs.size() == m_inputNodes[1]->GetNumOutputs());

    // copy values for 1st input
    std::copy(
        context.inputs.begin(), context.inputs.end(),
        context.outputs.begin());

    // copy values for 2nd input
    std::copy(
        context.secondaryInputs.begin(), context.secondaryInputs.end(),
        context.outputs.begin() + context.inputs.size());
}

void ConcatenationNode::Backpropagate(const Values& error, INodeContext& ctx, Gradients&) const
{
    Context& context = static_cast<Context&>(ctx);

    ASSERT(context.outputs.size() == error.size());
    ASSERT(context.outputs.size() == GetNumOutputs());
    ASSERT(context.inputError.size() == m_inputNodes[0]->GetNumOutputs());
    ASSERT(context.secondaryInputError.size() == m_inputNodes[1]->GetNumOutputs());

    // copy error for 1st input
    std::copy(
        error.begin(), error.begin() + context.inputs.size(),
        context.inputError.begin());

    // copy error for 2nd input
    std::copy(
        error.begin() + context.inputs.size(), error.end(),
        context.secondaryInputError.begin());
}

} // namespace nn
