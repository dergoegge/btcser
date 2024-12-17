#include "btcser_mutator.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Global mutator instance
static Mutator *g_mutator = nullptr;

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
  const char *descriptor = R"(
        Test {
            u32,           # version
            vec<u8>,       # payload
            bytes<32>      # hash
        }
    )";
  g_mutator = new Mutator(descriptor, "Test");
  return 0;
}

extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                          size_t MaxSize, unsigned int Seed) {
  if (!g_mutator || Size == 0)
    return 0;

  // Get mutated data
  MutatedBuffer mutated = g_mutator->mutate(Data, Size, Seed);
  if (!mutated.buffer)
    return 0;

  // Copy result if it fits
  size_t copy_size = std::min(MaxSize, (size_t)mutated.len);
  std::memcpy(Data, mutated.buffer, copy_size);
  return copy_size;
}

extern "C" size_t LLVMFuzzerCustomCrossOver(const uint8_t *Data1, size_t Size1,
                                            const uint8_t *Data2, size_t Size2,
                                            uint8_t *Out, size_t MaxOutSize,
                                            unsigned int Seed) {
  if (!g_mutator)
    return 0;

  // Perform cross-over
  MutatedBuffer crossed =
      g_mutator->cross_over(Data1, Size1, Data2, Size2, Seed);
  if (!crossed.buffer)
    return 0;

  // Copy result if it fits
  size_t copy_size = std::min(MaxOutSize, (size_t)crossed.len);
  std::memcpy(Out, crossed.buffer, copy_size);
  return copy_size;
}

// The actual fuzz target that tests the input
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  // Print data in hex
  printf("===\n");
  for (size_t i = 0; i < Size; i++) {
    printf("%02x", Data[i]);
  }
  printf("\n");

  return 0;
}
