# OpenCV snippets and useful patterns

For Linux and Macos builds hermetic OpenCV. 

For Windows requires pre-build libraries and `config=windows`
in the terminal and in `.bazelproject` add 

```
build_flags:
  --config=windows

test_flags:
  --config=windows
```
