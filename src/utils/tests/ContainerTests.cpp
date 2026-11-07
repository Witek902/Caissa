#include "TestCommon.hpp"
#include "../../backend/containers/DynArray.hpp"
#include "../../backend/containers/FixedString.hpp"
#include "../../backend/containers/Sort.hpp"
#include "../../backend/containers/String.hpp"

#include <string>

// counts live instances to check that DynArray constructs and destroys every element exactly once
struct LifetimeCounter
{
    static inline int32_t numAlive = 0;
    int32_t value = 0;

    LifetimeCounter() { numAlive++; }
    LifetimeCounter(int32_t v) : value(v) { numAlive++; }
    LifetimeCounter(const LifetimeCounter& other) : value(other.value) { numAlive++; }
    LifetimeCounter(LifetimeCounter&& other) noexcept : value(other.value) { other.value = -1; numAlive++; }
    LifetimeCounter& operator = (const LifetimeCounter& other) = default;
    LifetimeCounter& operator = (LifetimeCounter&& other) noexcept { value = other.value; other.value = -1; return *this; }
    ~LifetimeCounter() { numAlive--; }
    bool operator == (const LifetimeCounter& other) const { return value == other.value; }
};

void RunContainerTests()
{
    std::cout << "Running container tests..." << std::endl;

    // DynArray basics
    {
        DynArray<int32_t> a;
        TEST_EXPECT(a.Empty());
        TEST_EXPECT(a.Size() == 0);
        TEST_EXPECT(a.begin() == a.end());

        for (int32_t i = 0; i < 1000; ++i) a.PushBack(i);
        TEST_EXPECT(a.Size() == 1000);
        TEST_EXPECT(a.Capacity() >= 1000);
        TEST_EXPECT(a.Front() == 0);
        TEST_EXPECT(a.Back() == 999);
        TEST_EXPECT(a[500] == 500);

        int64_t sum = 0;
        for (const int32_t v : a) sum += v;
        TEST_EXPECT(sum == 999 * 1000 / 2);

        TEST_EXPECT(a.Find(123) == 123);
        TEST_EXPECT(a.Find(1000) == UINT32_MAX);
        TEST_EXPECT(a.Contains(999));
        TEST_EXPECT(!a.Contains(-1));

        a.PopBack();
        TEST_EXPECT(a.Size() == 999);
        TEST_EXPECT(a.Back() == 998);

        const uint32_t capacity = a.Capacity();
        a.Clear();
        TEST_EXPECT(a.Empty());
        TEST_EXPECT(a.Capacity() == capacity);
    }

    // DynArray resize value-initializes new elements
    {
        DynArray<uint64_t> a(5);
        TEST_EXPECT(a.Size() == 5);
        for (const uint64_t v : a) TEST_EXPECT(v == 0);

        a.Resize(8, 7);
        TEST_EXPECT(a.Size() == 8);
        TEST_EXPECT(a[4] == 0);
        TEST_EXPECT(a[5] == 7 && a[7] == 7);

        a.Resize(2);
        TEST_EXPECT(a.Size() == 2);
    }

    // DynArray erase, insert, assign keep the order
    {
        const int32_t values[] = { 0, 1, 2, 3, 4, 5 };
        DynArray<int32_t> a(values, 6);
        a.Erase(1, 3);
        TEST_EXPECT(a.Size() == 4);
        TEST_EXPECT(a[0] == 0 && a[1] == 3 && a[2] == 4 && a[3] == 5);

        a.Erase(3);
        TEST_EXPECT(a.Size() == 3 && a.Back() == 4);

        a.Insert(1, 42);
        TEST_EXPECT(a.Size() == 4);
        TEST_EXPECT(a[0] == 0 && a[1] == 42 && a[2] == 3 && a[3] == 4);

        a.Insert(4, 7);
        TEST_EXPECT(a.Back() == 7);

        a.Assign(values, 2);
        TEST_EXPECT(a.Size() == 2 && a[1] == 1);
    }

    // DynArray copy, move and comparison
    {
        DynArray<int32_t> a;
        for (int32_t i = 0; i < 10; ++i) a.PushBack(i);

        DynArray<int32_t> b = a;
        TEST_EXPECT(a == b);
        b[0] = 100;
        TEST_EXPECT(a[0] == 0);
        TEST_EXPECT(a != b);

        DynArray<int32_t> c = MoveTemp(b);
        TEST_EXPECT(b.Empty());
        TEST_EXPECT(c.Size() == 10 && c[0] == 100);

        b = c;
        TEST_EXPECT(b == c);
        a = MoveTemp(c);
        TEST_EXPECT(c.Empty());
        TEST_EXPECT(a == b);
    }

    // DynArray pushing its own element while reallocating
    {
        DynArray<LifetimeCounter> a;
        a.PushBack(LifetimeCounter(5));
        while (a.Size() < a.Capacity()) a.PushBack(LifetimeCounter(1));
        a.PushBack(a[0]);
        TEST_EXPECT(a.Back().value == 5);
        TEST_EXPECT(a.Front().value == 5);
    }
    TEST_EXPECT(LifetimeCounter::numAlive == 0);

    // DynArray element lifetimes
    {
        DynArray<LifetimeCounter> a;
        for (int32_t i = 0; i < 100; ++i) a.EmplaceBack(i);
        TEST_EXPECT(LifetimeCounter::numAlive == 100);
        a.Erase(10, 20);
        TEST_EXPECT(LifetimeCounter::numAlive == 90);
        TEST_EXPECT(a[10].value == 20);
        a.Resize(50);
        TEST_EXPECT(LifetimeCounter::numAlive == 50);
        a.Resize(60);
        TEST_EXPECT(LifetimeCounter::numAlive == 60);
        {
            DynArray<LifetimeCounter> b = a;
            TEST_EXPECT(LifetimeCounter::numAlive == 120);
        }
        TEST_EXPECT(LifetimeCounter::numAlive == 60);
        a.Clear();
        TEST_EXPECT(LifetimeCounter::numAlive == 0);
    }
    TEST_EXPECT(LifetimeCounter::numAlive == 0);

    // FixedString
    {
        FixedString<32> s;
        TEST_EXPECT(s.Empty());
        TEST_EXPECT(s == "");
        s.Append("e2");
        s.Append('e');
        s.Append('4');
        TEST_EXPECT(s == "e2e4");
        TEST_EXPECT(s.Length() == 4);
        TEST_EXPECT(s[3] == '4');
        s.AppendFormat(" %d %s", 42, "x");
        TEST_EXPECT(s == "e2e4 42 x");
        TEST_EXPECT(strcmp(s.CStr(), "e2e4 42 x") == 0);

        const FixedString<16> t = "e2e4 42 x";
        TEST_EXPECT(s == t);
        TEST_EXPECT(!(s != t));
        TEST_EXPECT(t != "e2e4");

        s.Clear();
        TEST_EXPECT(s.Empty() && s == "");

        // implicit conversion keeps std::string interop working
        const std::string str(t);
        TEST_EXPECT(str == "e2e4 42 x");
    }

    // String
    {
        String s;
        TEST_EXPECT(s.Empty());
        TEST_EXPECT(s.Length() == 0);
        TEST_EXPECT(strcmp(s.CStr(), "") == 0);

        s.Append("abc");
        s.Append('d');
        s.Append("efgh", 2);
        TEST_EXPECT(s == "abcdef");
        TEST_EXPECT(s.Length() == 6);

        for (uint32_t i = 0; i < 1000; ++i) s.AppendFormat("%04u", i);
        TEST_EXPECT(s.Length() == 6 + 4000);
        TEST_EXPECT(s[6] == '0' && s[s.Length() - 1] == '9');
        TEST_EXPECT(strncmp(s.CStr() + 6 + 4 * 999, "0999", 4) == 0);

        String copy = s;
        TEST_EXPECT(copy == s);
        copy.Append('!');
        TEST_EXPECT(copy != s);

        s.Clear();
        TEST_EXPECT(s.Empty() && s == "");
    }

    // Sort
    {
        const auto isSorted = [](const DynArray<uint32_t>& a)
        {
            for (uint32_t i = 1; i < a.Size(); ++i)
                if (a[i - 1] > a[i]) return false;
            return true;
        };
        const auto less = [](uint32_t a, uint32_t b) { return a < b; };

        for (const uint32_t size : { 0u, 1u, 2u, 3u, 16u, 17u, 100u, 5000u })
        {
            uint64_t state = 0x123456789ull + size;
            DynArray<uint32_t> a;
            uint64_t sum = 0;
            for (uint32_t i = 0; i < size; ++i)
            {
                state ^= state << 13; state ^= state >> 7; state ^= state << 17;
                a.PushBack(static_cast<uint32_t>(state % 1000));
                sum += a.Back();
            }

            Sort(a.begin(), a.end(), less);
            TEST_EXPECT(isSorted(a));

            uint64_t sortedSum = 0;
            for (const uint32_t v : a) sortedSum += v;
            TEST_EXPECT(sortedSum == sum);
        }

        // all equal, already sorted and reversed input
        {
            DynArray<uint32_t> a(1000, 7);
            Sort(a.begin(), a.end(), less);
            TEST_EXPECT(isSorted(a));

            for (uint32_t i = 0; i < a.Size(); ++i) a[i] = i;
            Sort(a.begin(), a.end(), less);
            TEST_EXPECT(isSorted(a) && a.Back() == 999);

            for (uint32_t i = 0; i < a.Size(); ++i) a[i] = 999 - i;
            Sort(a.begin(), a.end(), less);
            TEST_EXPECT(isSorted(a) && a.Front() == 0);
        }

        // descending order
        {
            uint32_t values[] = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
            Sort(values, values + 11, [](uint32_t a, uint32_t b) { return a > b; });
            for (uint32_t i = 1; i < 11; ++i) TEST_EXPECT(values[i - 1] >= values[i]);
            TEST_EXPECT(values[0] == 9 && values[10] == 1);
        }
    }
}
