#pragma once

#include "CombiningNode.hpp"

namespace nn {

// sum outputs of two nodes
class SumNode : public ICombiningNode
{
public:
    SumNode(const NodePtr& previousNodeA, const NodePtr& previousNodeB);
    virtual void Run(INodeContext& ctx) const override;
    virtual void Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const override;
};

} // namespace nn
