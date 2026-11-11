#pragma once

#include "../../backend/Common.hpp"

#include <cmath>
#include <iostream>

#define TEST_EXPECT(x)     if (!(x)) { std::cout << "Test failed: " << #x << " at " << __FILE__ << ":" << __LINE__ << std::endl; DEBUG_BREAK(); }

#define TEST_EXPECT_EQ(a, b)     if ((a) != (b)) { std::cout << "Test failed: " << #a << " == " << #b << " (got " << (a) << " != " << (b) << ") at " << __FILE__ << ":" << __LINE__ << std::endl; DEBUG_BREAK(); }

#define TEST_EXPECT_NEAR(a, b, epsilon)     if (std::abs((a) - (b)) > (epsilon)) { std::cout << "Test failed: " << #a << " near " << #b << " (got " << (a) << " != " << (b) << ") at " << __FILE__ << ":" << __LINE__ << std::endl; DEBUG_BREAK(); }

namespace threadpool { class TaskBuilder; }

void RunContainerTests();
void RunThreadTests();
void RunBitboardTests();
void RunBitboardAdvancedTests();
void RunBitOperationTests();
void RunBitboardTransformTests();
void RunSlidingAttackTests();
void RunRayTests();
void RunPieceAttackDetailTests();
void RunSquareUtilityTests();
void RunPositionTests();
void RunPositionAdvancedTests();
void RunScoreTests();
void RunMoveListTests();
void RunTranspositionTableTests();
void RunTimeTests();
void RunTimeManagerTests();
void RunEvalTests();
void RunPackedPositionTests();
void RunGameTests();
void RunPgnParserTests();
void RunPerftTests(threadpool::TaskBuilder& taskBuilder, bool deep);
void RunSearchTests();
