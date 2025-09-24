// Miscellaneous function to and fro protos
#ifndef PROTO_UTILS_H
#define PROTO_UTILS_H
#include <google/protobuf/text_format.h>
#include <fstream>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/message.h"
#include "transformations/calibration.h"
#include "transformations/proto/calibration_data.pb.h"

namespace opencv_snippets {

template <typename ProtoType>
absl::StatusOr<ProtoType> LoadFromTextProtoFile(absl::string_view file_path) {
  static_assert(std::is_base_of_v<google::protobuf::Message, ProtoType>,
                "ProtoType must be a protobuf message type");

  std::ifstream file(file_path.data());
  if (!file.is_open()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to open file: ", file_path));
  }

  std::string text_proto((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

  if (text_proto.empty()) {
    return absl::InvalidArgumentError(absl::StrCat("Empty file: ", file_path));
  }

  ProtoType proto;
  if (!google::protobuf::TextFormat::ParseFromString(text_proto, &proto)) {
    return absl::InternalError(
        absl::StrCat("Failed to parse proto message from file: ", file_path));
  }

  return proto;
}

// Converts proto into struct
IntrinsicCalibration ConvertIntrinsicCalibrationFromProto(
    const proto::IntrinsicCalibration& proto);

}  // namespace opencv_snippets

#endif  // PROTO_UTILS_H
