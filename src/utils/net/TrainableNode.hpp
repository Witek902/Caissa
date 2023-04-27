#pragma once

#include "Node.hpp"
#include "WeightsStorage.hpp"

namespace nn {

struct WeightsStorage;

// Base class for all trainable node types
// Trainable nodes have weights storage attached that can be trained
class ITrainableNode : public INode
{
public:
    ITrainableNode(const NodePtr& previousNode, const WeightsStoragePtr& weightsStorage,
        uint32_t inputSize, uint32_t outputSize, uint32_t numVariants = 1);

    virtual bool IsTrainable() const override { return true; }

    WeightsStorage* GetWeightsStorage() const { return m_weightsStorage.get(); }

protected:
    WeightsStoragePtr m_weightsStorage;
    NodePtr m_previousNode;
};

} // namespace nn
