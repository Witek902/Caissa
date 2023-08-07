#include "SumNode.hpp"

namespace nn {

OperatorNode::OperatorNode(const NodePtr& previousNodeA, const NodePtr& previousNodeB, Operator op)
    : ICombiningNode(previousNodeA->GetNumOutputs(), previousNodeA->GetNumOutputs())
    , m_operator(op)
{
    ASSERT(previousNodeA->GetNumOutputs() == previousNodeB->GetNumOutputs());

    m_inputNodes[0] = previousNodeA.get();
    m_inputNodes[1] = previousNodeB.get();
}

void OperatorNode::Run(INodeContext& ctx) const
{
    Context& context = static_cast<Context&>(ctx);

    ASSERT(context.outputs.size() == m_numOutputs);
    ASSERT(context.inputs.size() == m_inputNodes[0]->GetNumOutputs());
    ASSERT(context.secondaryInputs.size() == m_inputNodes[1]->GetNumOutputs());

    switch (m_operator)
    {
    case Operator::Sum:
        for (uint32_t i = 0; i < context.inputs.size(); ++i)
        {
            context.outputs[i] = context.inputs[i] + context.secondaryInputs[i];
        }
        break;
    case Operator::Product:
        for (uint32_t i = 0; i < context.inputs.size(); ++i)
        {
            context.outputs[i] = context.inputs[i] * context.secondaryInputs[i];
        }
        break;
    }
}

void OperatorNode::Backpropagate(const Values& error, INodeContext& ctx, Gradients&) const
{
    Context& context = static_cast<Context&>(ctx);

    ASSERT(context.outputs.size() == error.size());
    ASSERT(context.outputs.size() == GetNumOutputs());
    ASSERT(context.inputError.size() == m_inputNodes[0]->GetNumOutputs());
    ASSERT(context.secondaryInputError.size() == m_inputNodes[1]->GetNumOutputs());

    switch (m_operator)
    {
    case Operator::Sum:
        for (uint32_t i = 0; i < context.outputs.size(); ++i)
        {
            context.inputError[i] = error[i];
            context.secondaryInputError[i] = error[i];
        }
        break;
    case Operator::Product:
        for (uint32_t i = 0; i < context.outputs.size(); ++i)
        {
            context.inputError[i] = error[i] * context.secondaryInputs[i];
            context.secondaryInputError[i] = error[i] * context.inputs[i];
        }
        break;
    }
}

} // namespace nn
