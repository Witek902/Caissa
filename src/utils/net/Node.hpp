#pragma once

#include "Common.hpp"

#include <span>

namespace nn {

// how many nodes in the network can be input nodes
static constexpr uint32_t MaxInputNodes = 2;

struct Gradients;

enum class InputMode : uint8_t
{
    Unknown,
    Full,           // full list of inputs as floats
    Sparse,         // list of sparse inputs (as floats)
    SparseBinary,   // list of sparse binary inputs (active feature is always 1)
};

enum class OutputMode : uint8_t
{
    Unknown,
    Single,
    Full,
    // TODO sparse
};

class INodeContext
{
public:
    INodeContext(uint32_t numOutputs) : outputs(numOutputs) {}

    virtual ~INodeContext()
    {

    }

    std::span<const float> inputs;
    Values outputs;                     // saved by INode::Run()
    Values inputError;                  // saved by INode::Backpropagate()
};

// Base class for all node types
class INode
{
public:
    INode(uint32_t numInputs, uint32_t numOutputs);
    virtual ~INode() = default;

    virtual INodeContext* CreateContext() = 0;
    virtual void Run(INodeContext& ctx) const = 0;
    virtual void Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const = 0;

    virtual bool IsTrainable() const { return false; }
    virtual bool IsInputNode() const { return false; }
    virtual bool IsConcatenation() const { return false; }

    virtual InputMode GetInputMode() const { return InputMode::Unknown; }

    INLINE uint32_t GetNumInputs() const { return m_numInputs; }
    INLINE uint32_t GetNumOutputs() const { return m_numOutputs; }

protected:
    uint32_t m_numInputs;
    uint32_t m_numOutputs;
};

using NodePtr = std::shared_ptr<INode>;
using NodeContextPtr = std::unique_ptr<INodeContext>;

} // namespace nn
