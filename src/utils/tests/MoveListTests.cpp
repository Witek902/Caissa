#include "TestCommon.hpp"
#include "../../backend/MoveList.hpp"

void RunMoveListTests()
{
    std::cout << "Running MoveList tests..." << std::endl;

    MoveList list;
    
    TEST_EXPECT(list.Size() == 0);
    TEST_EXPECT(!list.HasMove(Move::Invalid()));
    
    // Push moves
    Move move1 = Move::Make(Square_e2, Square_e4, Piece::Pawn);
    Move move2 = Move::Make(Square_d2, Square_d4, Piece::Pawn);
    Move move3 = Move::Make(Square_g1, Square_f3, Piece::Knight);
    
    list.Push(move1);
    TEST_EXPECT(list.Size() == 1);
    TEST_EXPECT(list.HasMove(move1));
    TEST_EXPECT(!list.HasMove(move2));
    
    list.Push(move2);
    TEST_EXPECT(list.Size() == 2);
    TEST_EXPECT(list.HasMove(move1));
    TEST_EXPECT(list.HasMove(move2));
    
    list.Push(move3);
    TEST_EXPECT(list.Size() == 3);
    
    // GetMove
    TEST_EXPECT(list.GetMove(0) == move1);
    TEST_EXPECT(list.GetMove(1) == move2);
    TEST_EXPECT(list.GetMove(2) == move3);
    
    // RemoveMove
    list.RemoveMove(move2);
    TEST_EXPECT(list.Size() == 2);
    TEST_EXPECT(list.HasMove(move1));
    TEST_EXPECT(!list.HasMove(move2));
    TEST_EXPECT(list.HasMove(move3));
    
    // RemoveByIndex
    list.RemoveByIndex(0);
    TEST_EXPECT(list.Size() == 1);
    TEST_EXPECT(!list.HasMove(move1));
    TEST_EXPECT(list.HasMove(move3));
    
    // Clear
    list.Clear();
    TEST_EXPECT(list.Size() == 0);
    TEST_EXPECT(!list.HasMove(move3));
    
    // PackedMove operations
    {
        MoveList packedList;
        PackedMove pm1 = PackedMove(Square_e2, Square_e4);
        PackedMove pm2 = PackedMove(Square_d2, Square_d4);
        
        Move m1 = Move::Make(Square_e2, Square_e4, Piece::Pawn);
        Move m2 = Move::Make(Square_d2, Square_d4, Piece::Pawn);
        
        packedList.Push(m1);
        packedList.Push(m2);
        
        TEST_EXPECT(packedList.HasMove(pm1));
        TEST_EXPECT(packedList.HasMove(pm2));
        
        packedList.RemoveMove(pm1);
        TEST_EXPECT(!packedList.HasMove(pm1));
        TEST_EXPECT(packedList.HasMove(pm2));
    }
}
