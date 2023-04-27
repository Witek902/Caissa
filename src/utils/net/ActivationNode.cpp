#include "ActivationNode.hpp"

namespace nn {

INLINE static float ApplyActivationFunction(float x, ActivationFunction func)
{
    switch (func)
    {
    case ActivationFunction::ReLU:      return ReLU(x);
    case ActivationFunction::CReLU:     return CReLU(x);
    case ActivationFunction::SqrCReLU:  return SqrCReLU(x);
    case ActivationFunction::Sigmoid:   return Sigmoid(x);
    }
    return x;
}

INLINE static float GetActivationFunctionDerivative(float x, ActivationFunction func)
{
    switch (func)
    {
    case ActivationFunction::ReLU:      return ReLUDerivative(x);
    case ActivationFunction::CReLU:     return CReLUDerivative(x);
    case ActivationFunction::SqrCReLU:  return SqrCReLUDerivative(x);
    case ActivationFunction::Sigmoid:   return SigmoidDerivative(x);
    }
    return 1.0f;
}


ActivationNode::ActivationNode(const NodePtr& previousNode, ActivationFunction func)
    : INode(previousNode->GetNumOutputs(), previousNode->GetNumOutputs())
    , mActivationFunc(func)
{
}

void ActivationNode::Run(INodeContext& ctx) const
{
    ASSERT(ctx.outputs.size() == numOutputs);
    ASSERT(ctx.inputs.size() == numInputs);

#ifndef CONFIGURATION_FINAL
    for (size_t i = 0; i < numOutputs; i++)
    {
        const float x = ctx.inputs[i];
        ASSERT(!std::isnan(x));
        ASSERT(fabsf(x) < 10000.0f);
    }
#endif // CONFIGURATION_FINAL

    size_t i = 0;

#ifdef USE_AVX
    float* outputsPtr = ctx.outputs.data();
    const float* valuesPtr = ctx.inputs.data();
    if (mActivationFunc == ActivationFunction::ReLU)
    {
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(outputsPtr + i, ReLU(_mm256_load_ps(valuesPtr + i)));
        }
    }
    else if (mActivationFunc == ActivationFunction::CReLU)
    {
        for (; i + 8 <= numOutputs; i += 8)
        {
            _mm256_store_ps(outputsPtr + i, CReLU(_mm256_load_ps(valuesPtr + i)));
        }
    }
#endif // USE_AVX

    for (; i < numOutputs; i++)
    {
        ctx.outputs[i] = ApplyActivationFunction(ctx.inputs[i], mActivationFunc);
    }
}

void ActivationNode::Backpropagate(const Values& error, INodeContext& ctx, Gradients&) const
{
    ASSERT(ctx.outputs.size() == error.size());
    ASSERT(ctx.outputs.size() == GetNumOutputs());
    ASSERT(ctx.inputError.size() == GetNumInputs());

    size_t i = 0;
#ifdef USE_AVX
    const float* errorsPtr = error.data();
    const float* valuesPtr = ctx.inputs.data();
    if (mActivationFunc == ActivationFunction::ReLU)
    {
        for (; i + 8 <= numOutputs; i += 8)
            _mm256_store_ps(ctx.inputError.data() + i,
                ReLUDerivative(_mm256_load_ps(valuesPtr + i), _mm256_load_ps(errorsPtr + i)));
    }
    else if (mActivationFunc == ActivationFunction::CReLU)
    {
        for (; i + 8 <= numOutputs; i += 8)
            _mm256_store_ps(ctx.inputError.data() + i,
                CReLUDerivative(_mm256_load_ps(valuesPtr + i), _mm256_load_ps(errorsPtr + i)));
    }
#endif // USE_AVX
    for (; i < numOutputs; i++)
    {
        ctx.inputError[i] = error[i] * GetActivationFunctionDerivative(ctx.inputs[i], mActivationFunc);
    }
}

} // namespace nn
