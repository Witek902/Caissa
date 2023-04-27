#include "TrainableNode.hpp"
#include "../minitrace/minitrace.h"

#include <random>

namespace nn {

ITrainableNode::ITrainableNode(const NodePtr& previousNode, const WeightsStoragePtr& weightsStorage, uint32_t inputSize, uint32_t outputSize, uint32_t numVariants)
    : INode(inputSize, outputSize)
    , m_previousNode(previousNode)
    , m_weightsStorage(weightsStorage)
{
    ASSERT(numVariants > 0);

    // TODO variants
    UNUSED(numVariants);

    ASSERT(m_weightsStorage->m_inputSize == inputSize);
    ASSERT(m_weightsStorage->m_outputSize == outputSize);

    if (m_previousNode)
    {
        ASSERT(m_previousNode->GetNumOutputs() == inputSize);
    }
}

} // namespace nn
