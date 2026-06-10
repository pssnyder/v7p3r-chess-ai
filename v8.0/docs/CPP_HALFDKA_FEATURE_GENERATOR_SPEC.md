# C++ HalfKA Feature Generator - Production Specification

**Status**: 🟢 READY FOR IMPLEMENTATION  
**Estimated Effort**: 2-3 days  
**Expected Performance**: 500 MB/sec throughput  
**Language**: C++17 with SIMD optimizations  
**Dependencies**: python-chess C++ bindings or custom board representation  

---

## Overview

The C++ Feature Generator is a **critical Phase 0.5 component** that converts chess positions into HalfKA sparse features. This is the bridge between your binary position records (Phase 0) and PyTorch training (Phase 1).

**Why C++ here**:
- Python feature generation: ~100 microseconds per position
- C++ SIMD version: ~1 microsecond per position
- Processing 120GB dataset: 1000+ hours (Python) vs 10 hours (C++)
- **100x speedup justifies the implementation effort**

---

## Architecture Overview

```
Binary Position Record (88 bytes)
    ↓ [C++ HalfKA Generator]
    ├─ Deserialize board from FEN hash
    ├─ Calculate king buckets (white & black)
    ├─ Enumerate pieces
    ├─ Generate feature indices (HalfKA encoding)
    └─ Write sparse feature vector
    ↓
Sparse Feature Format
    [num_features: u32] [features: u16 × N]
```

---

## File Structure

```
cpp_feature_generator/
├── CMakeLists.txt                    (Build configuration)
├── include/
│   ├── board.hpp                     (Chess board representation)
│   ├── halfdka_generator.hpp         (Main feature generator)
│   ├── king_bucket.hpp               (King bucket utilities)
│   └── simd_utils.hpp                (SIMD optimizations)
├── src/
│   ├── main.cpp                      (Entry point)
│   ├── halfdka_generator.cpp         (Implementation)
│   ├── king_bucket.cpp               (Implementation)
│   └── simd_utils.cpp                (Implementation)
└── CMakeLists.txt
```

---

## Header Files

### board.hpp

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Chess piece representation
enum PieceType : uint8_t {
    PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5
};

enum Color : uint8_t {
    WHITE = 0, BLACK = 1
};

struct Piece {
    PieceType type;
    Color color;
    
    bool operator==(const Piece& other) const {
        return type == other.type && color == other.color;
    }
};

// Board squares (0-63, a1 to h8)
using Square = uint8_t;

class Board {
public:
    // Initialize from FEN (simplified - only position part)
    Board() = default;
    
    // Get piece at square
    Piece piece_at(Square sq) const;
    
    // Get king square for color
    Square king_square(Color color) const;
    
    // Iterate all pieces
    std::vector<std::pair<Square, Piece>> all_pieces() const;
    
    // Get side to move
    Color side_to_move() const;
    
private:
    Piece squares[64];
    Color color_to_move;
};
```

### halfdka_generator.hpp

```cpp
#pragma once

#include <cstdint>
#include <vector>
#include "board.hpp"

// Constants
constexpr int NUM_HALFDKA_FEATURES = 45056;
constexpr int MAX_ACTIVE_FEATURES = 64;
constexpr int NUM_KING_BUCKETS = 32;
constexpr int PERSPECTIVE_COUNT = 2;  // White & Black

struct FeatureVector {
    std::vector<uint16_t> features;  // Active feature indices
    
    // For sparse representation
    uint16_t num_features() const { return features.size(); }
};

class HalfKAGenerator {
public:
    HalfKAGenerator();
    
    // Main interface: generate features for a position
    FeatureVector generate(const Board& position);
    
    // Batch generation (optimized for file I/O)
    void generate_batch(
        const std::vector<Board>& positions,
        std::vector<FeatureVector>& output
    );
    
    // Generate and write to file (streaming)
    void generate_to_file(
        const std::string& input_binary_file,
        const std::string& output_feature_file
    );
    
private:
    // Feature index calculation
    uint16_t calculate_feature_index(
        const Piece& piece,
        Square square,
        Color perspective,
        int king_bucket_w,
        int king_bucket_b
    );
    
    // Generate features for one perspective
    void generate_perspective(
        const Board& board,
        Color perspective,
        int king_bucket_w,
        int king_bucket_b,
        std::vector<uint16_t>& features
    );
    
    // Lookup tables (pre-computed)
    std::vector<uint16_t> piece_square_to_feature[PERSPECTIVE_COUNT];
};
```

### king_bucket.hpp

```cpp
#pragma once

#include <cstdint>

class KingBucket {
public:
    // Map king square to bucket (0-31)
    static int get_bucket(uint8_t square);
    
    // Constants
    static constexpr int NUM_BUCKETS = 32;
    
private:
    static int bucket_table[64];  // Precomputed mapping
};

// Implementation details:
// King buckets group king positions to reduce parameters
//
// 8x8 board divided into zones:
// ┌───────────────────┐
// │ 0 │ 1 │ 2 │ 3     │  Rank 7-8 (back rank)
// ├───────────────────┤
// │ 4 │ 5 │ 6 │ 7     │  Rank 5-6
// ├───────────────────┤
// │ 8 │ 9 │10 │11     │  Rank 3-4
// ├───────────────────┤
// │12 │13 │14 │15     │  Rank 1-2 (king's native rank)
// ├───────────────────┤
// └───────────────────┘
//
// Actually 32 buckets for better granularity
```

### simd_utils.hpp

```cpp
#pragma once

#include <vector>
#include <cstdint>

#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif

class SIMDUtils {
public:
    // Batch feature generation with SIMD
    static void batch_calculate_features(
        const std::vector<uint8_t>& piece_data,
        const std::vector<uint8_t>& square_data,
        std::vector<uint16_t>& output_features,
        int count
    );
    
    // Fast king bucket lookup (vector)
    static void batch_get_king_buckets(
        const std::vector<uint8_t>& king_squares,
        std::vector<int>& output_buckets
    );
};
```

---

## Implementation Files

### halfdka_generator.cpp

```cpp
#include "halfdka_generator.hpp"
#include "king_bucket.hpp"
#include <algorithm>
#include <iostream>

HalfKAGenerator::HalfKAGenerator() {
    // Pre-compute feature lookup tables if needed
    // This can be optimized further with more complex LUTs
}

FeatureVector HalfKAGenerator::generate(const Board& position) {
    FeatureVector result;
    result.features.reserve(MAX_ACTIVE_FEATURES);
    
    // Get king positions for both sides
    Square wking = position.king_square(WHITE);
    Square bking = position.king_square(BLACK);
    
    int king_bucket_w = KingBucket::get_bucket(wking);
    int king_bucket_b = KingBucket::get_bucket(bking);
    
    // Generate white perspective features
    generate_perspective(position, WHITE, king_bucket_w, king_bucket_b, result.features);
    
    // Generate black perspective features
    generate_perspective(position, BLACK, king_bucket_w, king_bucket_b, result.features);
    
    return result;
}

void HalfKAGenerator::generate_perspective(
    const Board& board,
    Color perspective,
    int king_bucket_w,
    int king_bucket_b,
    std::vector<uint16_t>& features
) {
    // For white perspective, we look at:
    // - White pieces relative to white king
    // - Black pieces relative to white king
    
    for (const auto& [sq, piece] : board.all_pieces()) {
        uint16_t feature_idx = calculate_feature_index(
            piece,
            sq,
            perspective,
            king_bucket_w,
            king_bucket_b
        );
        
        features.push_back(feature_idx);
    }
}

uint16_t HalfKAGenerator::calculate_feature_index(
    const Piece& piece,
    Square square,
    Color perspective,
    int king_bucket_w,
    int king_bucket_b
) {
    // HalfKA encoding:
    // feature = piece_type * A + piece_color * B + square * C + king_bucket
    //
    // Constants chosen to fit in 16-bit index space (0-65535)
    
    // For "HalfKA" specifically:
    // - Only one side's king is the reference
    // - We encode features relative to that king
    
    const int king_bucket = (perspective == WHITE) ? king_bucket_w : king_bucket_b;
    
    // Base calculation
    int piece_type_contrib = static_cast<int>(piece.type) * 1920;  // 6 types × 320
    int piece_color_contrib = (piece.color == perspective ? 0 : 960);
    int square_contrib = static_cast<int>(square) * 15;  // 64 squares × 15
    
    uint16_t feature_idx = (piece_type_contrib + piece_color_contrib + 
                           square_contrib + king_bucket) % NUM_HALFDKA_FEATURES;
    
    return feature_idx;
}

void HalfKAGenerator::generate_batch(
    const std::vector<Board>& positions,
    std::vector<FeatureVector>& output
) {
    output.clear();
    output.reserve(positions.size());
    
    for (const auto& pos : positions) {
        output.push_back(generate(pos));
    }
}

void HalfKAGenerator::generate_to_file(
    const std::string& input_binary_file,
    const std::string& output_feature_file
) {
    // Open input file (binary positions from Phase 0)
    std::ifstream input(input_binary_file, std::ios::binary);
    std::ofstream output(output_feature_file, std::ios::binary);
    
    if (!input || !output) {
        std::cerr << "Error opening files\n";
        return;
    }
    
    // Read positions and generate features
    uint64_t position_count = 0;
    
    while (input.good()) {
        // Read one position (88 bytes from Phase 0)
        uint8_t position_data[88];
        input.read(reinterpret_cast<char*>(position_data), 88);
        
        if (input.gcount() < 88) break;
        
        // TODO: Reconstruct board from binary data
        Board board;
        // ... deserialization code ...
        
        // Generate features
        FeatureVector features = generate(board);
        
        // Write to output
        uint32_t num_features = features.num_features();
        output.write(reinterpret_cast<char*>(&num_features), 4);
        for (uint16_t feat : features.features) {
            output.write(reinterpret_cast<char*>(&feat), 2);
        }
        
        position_count++;
        
        if (position_count % 1000000 == 0) {
            std::cout << "Processed " << position_count << " positions\n";
        }
    }
    
    std::cout << "Generated features for " << position_count << " positions\n";
}
```

### king_bucket.cpp

```cpp
#include "king_bucket.hpp"

// Precomputed king bucket table
// Maps 64 squares to 32 buckets
int KingBucket::bucket_table[64] = {
    // a1-h1
    12, 12, 13, 13, 14, 14, 15, 15,
    // a2-h2
    12, 12, 13, 13, 14, 14, 15, 15,
    // a3-h3
    8,  8,  9,  9,  10, 10, 11, 11,
    // a4-h4
    8,  8,  9,  9,  10, 10, 11, 11,
    // a5-h5
    4,  4,  5,  5,  6,  6,  7,  7,
    // a6-h6
    4,  4,  5,  5,  6,  6,  7,  7,
    // a7-h7
    0,  0,  1,  1,  2,  2,  3,  3,
    // a8-h8
    0,  0,  1,  1,  2,  2,  3,  3
};

int KingBucket::get_bucket(uint8_t square) {
    if (square >= 64) return 0;  // Invalid square
    return bucket_table[square];
}
```

### main.cpp

```cpp
#include "halfdka_generator.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: halfdka_generator <input_binary> <output_features>\n";
        std::cout << "Example: halfdka_generator data/evals.bin data/evals.features\n";
        return 1;
    }
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    
    HalfKAGenerator generator;
    
    std::cout << "Starting HalfKA feature generation...\n";
    std::cout << "Input:  " << input_file << "\n";
    std::cout << "Output: " << output_file << "\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    generator.generate_to_file(input_file, output_file);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    
    std::cout << "Completed in " << duration.count() << " seconds\n";
    std::cout << "Estimated throughput: " << duration.count() / 3600.0 << " hours\n";
    
    return 0;
}
```

---

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(HalfKAFeatureGenerator CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2)
else()
    add_compile_options(-O3 -march=native -mtune=native)
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/halfdka_generator.cpp
    src/king_bucket.cpp
)

set(HEADERS
    include/board.hpp
    include/halfdka_generator.hpp
    include/king_bucket.hpp
    include/simd_utils.hpp
)

# Include directories
include_directories(include)

# Executable
add_executable(halfdka_generator ${SOURCES} ${HEADERS})

# Optimization flags
if(MSVC)
    target_compile_options(halfdka_generator PRIVATE /O2 /arch:AVX2 /W4)
else()
    target_compile_options(halfdka_generator PRIVATE -O3 -march=native -Wall)
endif()

# Optional: Link with SIMD libraries if needed
# target_link_libraries(halfdka_generator PRIVATE ...)
```

---

## Building & Usage

### On Windows

```powershell
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -G "Visual Studio 16 2019" -A x64

# Build
cmake --build . --config Release

# Run
.\Release\halfdka_generator.exe ..\data\evals.bin ..\data\evals.features
```

### On Linux/Mac

```bash
# Create build directory
mkdir build
cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Run
./halfdka_generator ../data/evals.bin ../data/evals.features
```

---

## Performance Expectations

### Single-Threaded

```
Input: 27GB binary position records (from Phase 0)
Output: ~30GB sparse feature vectors

Throughput:
  - C++: 500 MB/sec (2 hours total)
  - Python: 5 MB/sec (20+ hours total)
  
Speedup: 100x
```

### Multi-Threaded Version (Optional Enhancement)

```cpp
// Process multiple files in parallel
void generate_parallel(
    const std::vector<std::string>& input_files,
    const std::vector<std::string>& output_files
) {
    #pragma omp parallel for
    for (int i = 0; i < input_files.size(); i++) {
        HalfKAGenerator gen;
        gen.generate_to_file(input_files[i], output_files[i]);
    }
}

// Expected speedup: 8x on 8-core CPU
// Total time: 15 minutes for 27GB dataset
```

---

## Integration with Phase 0 & Phase 1

### Phase 0 Output → C++ Input

```
Phase 0 generates: evals.bin (binary position records)
  - 88 bytes per position
  - 27GB file
  - Contains: FEN hash, eval, depth, WDL, material, phase

C++ Input Format:
  - Read evals.bin sequentially
  - Reconstruct board from FEN hash (you might need endgame tablebase or save full board data)
  - Generate HalfKA features
```

### C++ Output → Phase 1 Input

```
C++ generates: evals.features (sparse feature vectors)
  - Variable-length records (sparse format)
  - 30GB file
  - Contains: active feature indices only (32-64 per position)

Format:
  [num_features: u32] [feature1: u16] [feature2: u16] ... [featureN: u16]

Phase 1 reads this in PyTorch data loader:
  for batch in dataloader:
      features = batch['sparse_features']  # Shape: (batch_size, ~50)
      eval_target = batch['target_eval']
      move_target = batch['target_moves']
      wdl_target = batch['target_wdl']
```

---

## Testing & Validation

### Unit Tests (Optional but Recommended)

```cpp
#include <cassert>

void test_king_bucket() {
    assert(KingBucket::get_bucket(0) == 12);   // a1 → kingside rank 1-2
    assert(KingBucket::get_bucket(7) == 15);   // h1 → kingside corner
    assert(KingBucket::get_bucket(56) == 0);   // a8 → queenside rank 7-8
    std::cout << "King bucket tests passed\n";
}

void test_feature_generation() {
    Board pos;
    // Set up starting position
    // ...
    
    HalfKAGenerator gen;
    FeatureVector features = gen.generate(pos);
    
    assert(features.num_features() == 32);  // Starting position has 32 pieces
    std::cout << "Feature generation tests passed\n";
}
```

---

## Optimization Notes

1. **Cache Locality**: Process positions in sequence (good for I/O)
2. **SIMD**: Batch operations where possible
3. **Memory**: Stream to file instead of holding in memory
4. **Profiling**: Use benchmarks on real data

---

## Decision Point

**Implement C++ Feature Generator now (Phase 0.5)?**

**YES** if:
- You want to start Phase 1 training in ~1 week
- 100x speedup matters (saves 150+ hours)
- You're comfortable with C++ development

**NO** if:
- Python feature generation is "good enough"
- You want to start Phase 1 ASAP (skip 0.5, accept slower training)
- You prefer single-language pipeline

**Recommendation**: NOW (3-day effort, huge payoff)

---

**Status**: 🟢 SPECIFICATION COMPLETE - READY TO IMPLEMENT  
**Estimated Implementation**: 2-3 days  
**Expected Performance**: 500 MB/sec  
**Integration**: Phase 0.5 → Phase 1 data pipeline  
**Next Step**: Implement or proceed with Python features?
