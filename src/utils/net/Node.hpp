#pragma once

#include "Common.hpp"

#include <span>

namespace nn {

enum class InputMode : uint8_t
{
    Unknown,
    Full,           // full list of inputs as floats
    Sparse,         // list of sparse inputs (as floats)
    SparseBinary,   // list of sparse binary inputs (active feature is always 1)
};

enum class OutputMode : uint8_t
{
    Single,
    Array,
};

class INodeContext
{
public:
    INodeContext(uint32_t numOutputs) : outputs(numOutputs) {}

    virtual ~INodeContext() = default;

    std::span<const float> inputs;
    Values outputs;
    Values inputError; // used for learning
};

struct Gradients
{
    uint32_t            m_numInputs;
    uint32_t            m_numOutputs;
    Values              m_values;
    std::vector<bool>   m_dirty;

    void Init(uint32_t numInputs, uint32_t numOutputs);
    void Clear();
    void Accumulate(Gradients& rhs);
};

// Base class for all node types
class INode
{
public:
    INode(uint32_t inputSize, uint32_t outputSize);
    virtual ~INode() = default;

    virtual INodeContext* CreateContext() = 0;
    virtual void Run(INodeContext& ctx) const = 0;
    virtual void Backpropagate(const Values& error, INodeContext& ctx, Gradients& gradients) const = 0;

    virtual bool IsTrainable() const { return false; }

    virtual InputMode GetInputMode() const { return InputMode::Unknown; }

    uint32_t GetNumInputs() const { return numInputs; }
    uint32_t GetNumOutputs() const { return numOutputs; }

protected:
    uint32_t numInputs;
    uint32_t numOutputs;
};

using NodePtr = std::shared_ptr<INode>;
using NodeContextPtr = std::unique_ptr<INodeContext>;

} // namespace nn
