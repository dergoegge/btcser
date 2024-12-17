#ifndef BTCSER_MUTATOR_H
#define BTCSER_MUTATOR_H

#include <stdint.h>

#ifdef __cplusplus
#include <cassert>
#include <utility>

extern "C" {
#endif

typedef struct CMutator CMutator;

// Create a new btcser mutator
CMutator *btcser_mutator_new(const char *descriptor,
                             const char *descriptor_name);

// Free a btcser mutator
void btcser_mutator_free(CMutator *mutator);

// Perform a mutation on a Bitcoin serialized blob
void btcser_mutator_mutate(const CMutator *mutator, const uint8_t *data,
                           uint32_t data_len, uint64_t seed, uint8_t **out,
                           uint32_t *out_len);

// Perform a cross-over mutation between two Bitcoin serialized blobs
void btcser_mutator_cross_over(const CMutator *mutator, const uint8_t *data1,
                               uint32_t data1_len, const uint8_t *data2,
                               uint32_t data2_len, uint64_t seed, uint8_t **out,
                               uint32_t *out_len);

// Free a buffer allocated by the mutator (i.e. buffer returned throught the
// `out` params)
void btcser_mutator_free_buffer(uint8_t *buffer, uint32_t len);

#ifdef __cplusplus
}

/** RAII wrapper for buffers allocated by a btcser mutator */
class MutatedBuffer {
public:
  uint8_t *buffer{nullptr};
  uint32_t len{0};

  MutatedBuffer() = default;
  MutatedBuffer(MutatedBuffer &&) = default;
  MutatedBuffer &operator=(MutatedBuffer &&) = default;
  MutatedBuffer(const MutatedBuffer &other) = delete;
  MutatedBuffer &operator=(const MutatedBuffer &other) = delete;

  ~MutatedBuffer() { btcser_mutator_free_buffer(buffer, len); }
};

/**
 * RAII wrapper for a btcser mutator.
 *
 * It is responsible for creating and freeing an underlying `CMutator` instance.
 * It also provides methods to perform mutations and cross-overs.
 *
 * Example usage with LibFuzzer:
 *
 * ```c++
 * static Mutator *g_mutator = nullptr;
 *
 * extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
 *   const char *descriptor = R"(
 *       Test {
 *         u32
 *         vec<u8>,
 *         bytes<32>
 *       }
 *     )";
 *   g_mutator = new Mutator(descriptor, "Test");
 *   return 0;
 * }
 *
 * extern "C" size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
 *                                           size_t MaxSize, unsigned int Seed) {
 *   if (!g_mutator || Size == 0)
 *     return 0;
 *   MutatedBuffer mutated = g_mutator->mutate(Data, Size, Seed);
 *   if (!mutated.buffer)
 *     return 0;
 *   size_t copy_size = std::min(MaxSize, (size_t)mutated.len);
 *   std::memcpy(Data, mutated.buffer, copy_size);
 *   return copy_size;
 * }
 *
 * extern "C" size_t LLVMFuzzerCustomCrossOver(const uint8_t *Data1, size_t Size1,
 *                                             const uint8_t *Data2, size_t Size2,
 *                                             uint8_t *Out, size_t MaxOutSize,
 *                                             unsigned int Seed) {
 *   if (!g_mutator)
 *     return 0;
 *   MutatedBuffer crossed =
 *       g_mutator->cross_over(Data1, Size1, Data2, Size2, Seed);
 *   if (!crossed.buffer)
 *     return 0;
 *   size_t copy_size = std::min(MaxOutSize, (size_t)crossed.len);
 *   std::memcpy(Out, crossed.buffer, copy_size);
 *   return copy_size;
 * }
 *
 * int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
 *   // Data represents a Bitcoin serialized blob (matching the descriptor of
 *   // the mutator)
 *   ...
 * }
 * ```
 */
class Mutator {
  CMutator *m_mutator;

public:
  Mutator(const char *descriptor, const char *descriptor_name) {
    m_mutator = btcser_mutator_new(descriptor, descriptor_name);
  }
  ~Mutator() { btcser_mutator_free(m_mutator); }

  MutatedBuffer mutate(const uint8_t *data, uint32_t data_len, uint64_t seed) {
    MutatedBuffer out;
    assert(m_mutator);
    btcser_mutator_mutate(m_mutator, data, data_len, seed, &out.buffer,
                          &out.len);
    return out;
  }

  MutatedBuffer cross_over(const uint8_t *data1, uint32_t data1_len,
                           const uint8_t *data2, uint32_t data2_len,
                           uint64_t seed) {
    MutatedBuffer out;
    assert(m_mutator);
    btcser_mutator_cross_over(m_mutator, data1, data1_len, data2, data2_len,
                              seed, &out.buffer, &out.len);
    return out;
  }
};

#endif

#endif // BTCSER_MUTATOR_H

