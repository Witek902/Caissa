#include "TestCommon.hpp"
#include "../../backend/Game.hpp"
#include "../../backend/Time.hpp"
#include "../../backend/TimeManager.hpp"

void RunTimeTests()
{
    std::cout << "Running Time tests..." << std::endl;

    // TimePoint basic operations
    {
        TimePoint invalid = TimePoint::Invalid();
        TEST_EXPECT(!invalid.IsValid());
        
        TimePoint now = TimePoint::GetCurrent();
        TEST_EXPECT(now.IsValid());
        
        TimePoint fromSeconds = TimePoint::FromSeconds(1.5f);
        TEST_EXPECT(fromSeconds.IsValid());
        TEST_EXPECT_NEAR(fromSeconds.ToSeconds(), 1.5f, 0.01f);
    }
    
    // TimePoint arithmetic
    {
        TimePoint t1 = TimePoint::FromSeconds(2.0f);
        TimePoint t2 = TimePoint::FromSeconds(1.0f);
        
        TimePoint diff = t1 - t2;
        TEST_EXPECT_NEAR(diff.ToSeconds(), 1.0f, 0.01f);
        
        TimePoint sum = t1 + t2;
        TEST_EXPECT_NEAR(sum.ToSeconds(), 3.0f, 0.01f);
        
        TimePoint t3 = t1;
        t3 *= 2.0;
        TEST_EXPECT_NEAR(t3.ToSeconds(), 4.0f, 0.01f);
    }
    
    // TimePoint comparisons
    {
        TimePoint t1 = TimePoint::FromSeconds(1.0f);
        TimePoint t2 = TimePoint::FromSeconds(2.0f);
        
        TEST_EXPECT(t2 >= t1);
        TEST_EXPECT(t1 != t2);
        TEST_EXPECT(!(t1 >= t2));
    }
}

void RunTimeManagerTests()
{
    std::cout << "Running TimeManager tests..." << std::endl;

    Game game;
    game.Reset(Position(Position::InitPositionFEN));
    
    SearchLimits limits;
    TimeManagerInitData initData;
    initData.moveTime = INT32_MAX;
    initData.remainingTime = 60000; // 60 seconds
    initData.timeIncrement = 1000;   // 1 second increment
    initData.theirRemainingTime = 60000;
    initData.theirTimeIncrement = 1000;
    initData.movesToGo = UINT32_MAX;
    initData.moveOverhead = 50;
    initData.previousSearchHint = PreviousSearchHint::Unknown;
    
    InitTimeManager(game, initData, limits);
    TEST_EXPECT(limits.idealTimeBase.IsValid());
    TEST_EXPECT(limits.maxTime.IsValid());
    
    // Fixed move time
    {
        SearchLimits fixedLimits;
        TimeManagerInitData fixedData;
        fixedData.moveTime = 5000; // 5 seconds
        fixedData.remainingTime = INT32_MAX;
        fixedData.timeIncrement = 0;
        fixedData.theirRemainingTime = INT32_MAX;
        fixedData.theirTimeIncrement = 0;
        fixedData.movesToGo = UINT32_MAX;
        fixedData.moveOverhead = 0;
        
        InitTimeManager(game, fixedData, fixedLimits);
        TEST_EXPECT(fixedLimits.idealTimeBase.IsValid());
        TEST_EXPECT_NEAR(fixedLimits.idealTimeBase.ToSeconds(), 5.0f, 0.1f);
    }
}
