#ifndef BTCSER_MUTATOR_H
#define BTCSER_MUTATOR_H

#include <stdint.h>

#ifdef __cplusplus
#include <utility>

extern "C" {
#endif

typedef struct CMutator CMutator;

CMutator *btcser_mutator_new(const char *descriptor,
                             const char *descriptor_name);

void btcser_mutator_free(CMutator *mutator);

void btcser_mutator_mutate(const CMutator *mutator, const uint8_t *data,
                           uint32_t data_len, uint64_t seed, uint8_t **out,
                           uint32_t *out_len);
void btcser_mutator_cross_over(const CMutator *mutator, const uint8_t *data1,
                               uint32_t data1_len, const uint8_t *data2,
                               uint32_t data2_len, uint64_t seed, uint8_t **out,
                               uint32_t *out_len);

void btcser_mutator_free_buffer(uint8_t *buffer, uint32_t len);

#ifdef __cplusplus
}

class MutatedBuffer {
public:
  uint8_t *buffer;
  uint32_t len;

  MutatedBuffer() = default;
  MutatedBuffer(MutatedBuffer &&) = default;
  MutatedBuffer &operator=(MutatedBuffer &&) = default;
  MutatedBuffer(const MutatedBuffer &other) = delete;
  MutatedBuffer &operator=(const MutatedBuffer &other) = delete;

  ~MutatedBuffer() { btcser_mutator_free_buffer(buffer, len); }
};

class Mutator {
  CMutator *mutator;

public:
  Mutator(const char *descriptor, const char *descriptor_name) {
    mutator = btcser_mutator_new(descriptor, descriptor_name);
  }
  ~Mutator() { btcser_mutator_free(mutator); }
  MutatedBuffer mutate(const uint8_t *data, uint32_t data_len, uint64_t seed) {
    MutatedBuffer out;
    btcser_mutator_mutate(mutator, data, data_len, seed, &out.buffer, &out.len);
    return out;
  }
  MutatedBuffer cross_over(const uint8_t *data1, uint32_t data1_len,
                           const uint8_t *data2, uint32_t data2_len,
                           uint64_t seed) {
    MutatedBuffer out;
    btcser_mutator_cross_over(mutator, data1, data1_len, data2, data2_len, seed,
                              &out.buffer, &out.len);
    return out;
  }
};

#endif

#endif // BTCSER_MUTATOR_H
