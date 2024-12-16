*Note: This is a work in progress.*

# btcser

btcser is a descriptor language for Bitcoin's serialization format (used
heavily in Bitcoin Core for data storage as well as the Bitcoin p2p protocol).
It is designed to be simple and easy to write for those familiar with Bitcoin's
serialization format.

Let's look at a simple example:

```text
foo { u8, u64, bytes<8> }
```

In this example, a new type `foo` is defined which consists of three fields: a
`u8`, a `u64`, and a `bytes<8>`. `u8` is a 8-bit unsigned integer, `u64` is a
64-bit little-endian unsigned integer, and `bytes<8>` is an 8-byte byte array.
Using Bitcoin's serialization format, this type would always be represented as
17 bytes. The first byte represent the `u8` field, the next 8 bytes represent
the `u64` field, and the last 8 bytes represent the `bytes<8>` field.

## Built-in Types
Besides the three basic types used above, btcser also supports the following
build-in types:

- `bool`: a boolean type
- `u8`, `i8`: 8-bit unsigned and signed integers
- `u16`, `u32`, `u64`, `u256`: unsigned integers of various sizes
  (little-endian)
- `i16`, `i32`, `i64`, `i256`: signed integers of various sizes (little-endian)
- `U16`, `U32`, `U64`, `U256`: unsigned integers of various sizes (big-endian)
- `I16`, `I32`, `I64`, `I256`: signed integers of various sizes (big-endian)
- `vec<T>`: a vector of type `T`
- `slice<T, 'f'>`: a slice of type `T` with its length specified by a preceding
  field at index `f`
- `bytes<N>`: a fixed-size byte array of length `N`
- `cs64`: compact size encoding of up to 64-bit integers
- `varint`: variable length integer encoding (default mode)
- `varint+`: variable length integer encoding (signed non-negative mode)

## Type Composition

Let's look at another example:

```text
foo { u8, u64, bytes<8> }

bar { vec<foo> }
```

In this example, `bar` is a type that consists of a vector of `foo`s. In
serialized form `bar` is of variable length depending on the number of `foo`s
it contains. Note how the `foo` type is defined in the first line and then used
as part of `bar`'s definition. This is how we can compose complex types from
simpler ones. For example, here is the Bitcoin `headers` protocol message
expressed in btcser:

```text
header { i32,u256,u256,u32,u32,u32 }
empty_block { header,u8(0) }
headers { vec<empty_block> }
```

## Slices

The `slice<T, 'f'>` type is used to specify a list of same-typed elements with
the list's length specified by a preceding field at index `f`. For example:

```text
bar { cs64, slice<u64, '0'> }
```

In this example, `bar` is a type that consists of a `cs64` field and a
`slice<u64, '0'>` field. The `cs64` field is a compact size encoding of a
64-bit integer, and the `slice<u64, '0'>` field is a slice of `u64`s with its
length specified by the `cs64` field. Those familiar with Bitcoin's
serialization format will recognize that this is the way that variable length
arrays are encoded. In short, it is the same as `vec<u64>`.

## Constants

btcser also supports constants as part of type definitions. For example:

```text
bar { bytes<4>(0xdeadbeef), u8(0xff) }
```

In this example, the `bar` type consists of a `bytes<4>` constant and a `u8`
constant. The `bytes<4>` constant is a 4-byte byte array with the value
`0xdeadbeef`, and the `u8` constant is an 8-bit unsigned integer with the value
`0xff`.

## Alternatives

btcser also supports alternatives, which are used to express the idea that
types can have alternative serialization formats. For example:

```text
bar { u8(0x01), vec<u64> }
bar { u8(0x02), bytes<32> }

foo { bar }
```

In this example, `foo` holds a `bar` field which can be either a `u8(0x01)` and
a `vec<u64>` or a `u8(0x02)` and a `bytes<32>`. Any object of type `foo` has to
be prefixed with either a `u8(0x01)` or a `u8(0x02)`.

*See [examples/](btcser/examples/) for more examples.*

## Applications in Fuzzing

btcser is primarily intended to be used for fuzzing Bitcoin full-node software,
as serializable types are often part of their interfaces. It allows for
structure-aware fuzzing of interfaces accepting Bitcoin serialized data, as it
provides fuzz engines with structure and type information that is otherwise
invisible to them.

Without being aware of the serialization structure, fuzz engines often mutate
inputs in ways that cause deserialization to fail (since they only operate on
byte arrays). When attempting to test logic that comes after inputs are
deserialized in a target, then violating serialization rules is a waste of CPU
cycles.

Structure and type information are very useful for cross-over mutations between
different serialized objects of the same schema. Let's consider this example:

```
tx { ... } # fields omitted for brevity (this is how comments work in btcser)
header { i32,u256,u256,u32,u32,u32 }
block { header, vec<tx> }
```

A byte array fuzzer, attempting to perform cross-over mutations between `block`
objects would frequently invalidate serialization structure. It simply isn't
aware of the individual fields and their types.

With btcser's additional structure and type information, it can now perform
cross-over mutations between `block` objects without messing with the overall
structure. Type information helps crossing over values of the same type, e.g.
add a transaction from one block to another, copy the prev block hash from one
header to another, etc.
