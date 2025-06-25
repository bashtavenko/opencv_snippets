#ifndef UTIL_STATUS_MACROS_H_
#define UTIL_STATUS_MACROS_H_

#define RETURN_IF_ERROR(expr)           \
  do {                                  \
    const absl::Status status = (expr); \
    if (!status.ok()) return status;    \
  } while (0)

#endif //UTIL_STATUS_MACROS_H_
