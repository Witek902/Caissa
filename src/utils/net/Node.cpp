#include "Node.hpp"

#include <random>

namespace nn {

INode::INode(uint32_t numInputs, uint32_t numOutputs)
    : m_numInputs(numInputs)
    , m_numOutputs(numOutputs)
{
}

} // namespace nn
