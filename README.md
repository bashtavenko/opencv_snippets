# Bazel OpenCV snippets and useful patterns on Linux, Mac and Windows

For two years I have been using Bazel `cmake()` from `rules_foreign_cc` for hermetic OpenCV builds, but it turned out to
be a disaster for a few reasons:

1. `rules_foreign_cc` has no concept of building from two related repositories, main OpenCV and contrib. There are some
   workarounds, but it is a waste of time.
2. Bazel with `cmake()` misteriously recompiles OpenCV for uknown reasons.
3. It takes 5 minutes or more on beefy Linux machines to build OpenCV. 

It is much simpler to build whatever version is needed from scratch for different platforms.
For Windows it requires `config=windows` and in `.bazelproject` add

```
build_flags:
  --config=windows

test_flags:
  --config=windows
```
