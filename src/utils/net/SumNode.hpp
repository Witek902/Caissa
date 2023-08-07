#pragma once

#include "CombiningNode.hpp"

namespace nn {

enum class Operator
{
    Sum,
    Product,
};

// sum outputs of two nodes
class OperatorNode : public ICombiningNode
{
public:
    OperatorNode(const NodePtr& previousNodeA, const NodePtr& previousNodeB, Operator op = Operator::Sum);
    virtual void Run(INodeContext& ctx) const override;
    virtual void Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const override;
private:
    Operator m_operator;
};

} // namespace nn
