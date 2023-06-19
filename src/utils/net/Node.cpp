#include "Node.hpp"

#include <random>

static constexpr float c_activationEpsilon = 1.0e-10f;

namespace nn {

INode::INode(uint32_t numInputs, uint32_t numOutputs)
    : m_numInputs(numInputs)
    , m_numOutputs(numOutputs)
{
}

} // namespace nn
