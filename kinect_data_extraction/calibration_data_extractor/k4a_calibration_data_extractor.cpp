#include <k4a/k4a.h>
#include <k4arecord/playback.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
using json = nlohmann::json;

// JSON key names
const char* const kColorCameraCalibrationKeyName = "color_camera_calibration";
const char* const kColorResolutionKeyName = "color_resolution";
const char* const kDepthModeKeyName = "depth_mode";
const char* const kExtrinsicsKeyName = "extrinsics";
const char* const kIntrinsicsKeyName = "intrinsics";
const char* const kParametersKeyName = "parameters";
const char* const kParamKeyName = "param";
const char* const kRorationKeyName = "rotation";
const char* const kTranslationKeyName = "translation";
const char* const kDepthCameraCalibrationKeyName = "depth_camera_calibration";
const char* const kMetricRadiusKeyName = "metric_radius";
const char* const kResolutionHeightKeyName = "resolution_height";
const char* const kResolutionWidthKeyName = "resolution_width";
const char* const kTypeKeyName = "type";
const char* const kParamterCountKeyName = "parameter_count";

// =================================================================================
// Helper functions to convert k4a structs to JSON
// =================================================================================
static json convert_k4a_calibration_extrinsics(
    k4a_calibration_extrinsics_t extrinsics) {
  json extrinsics_data;
  extrinsics_data[kRorationKeyName] = extrinsics.rotation;
  extrinsics_data[kTranslationKeyName] = extrinsics.translation;
  return extrinsics_data;
}

static json convert_k4a_calibration_intrinsic_parameters(
    k4a_calibration_intrinsic_parameters_t params) {
  json param_data;

  // Copy individual parameters
  param_data[kParamKeyName]["cx"] = params.param.cx;
  param_data[kParamKeyName]["cy"] = params.param.cy;
  param_data[kParamKeyName]["fx"] = params.param.fx;
  param_data[kParamKeyName]["fy"] = params.param.fy;
  param_data[kParamKeyName]["k1"] = params.param.k1;
  param_data[kParamKeyName]["k2"] = params.param.k2;
  param_data[kParamKeyName]["k3"] = params.param.k3;
  param_data[kParamKeyName]["k4"] = params.param.k4;
  param_data[kParamKeyName]["k5"] = params.param.k5;
  param_data[kParamKeyName]["k6"] = params.param.k6;
  param_data[kParamKeyName]["codx"] = params.param.codx;
  param_data[kParamKeyName]["cody"] = params.param.cody;
  param_data[kParamKeyName]["p2"] = params.param.p2;
  param_data[kParamKeyName]["p1"] = params.param.p1;
  param_data[kParamKeyName]["metric_radius"] = params.param.metric_radius;

  // Copy Vector reporesentation
  param_data["v"] = params.v;

  return param_data;
}

std::string convert_k4a_calibration_model_type_to_string(
    k4a_calibration_model_type_t calib_mode_type) {
  switch (calib_mode_type) {
    case K4A_CALIBRATION_LENS_DISTORTION_MODEL_UNKNOWN:
      return "K4A_CALIBRATION_LENS_DISTORTION_MODEL_UNKNOWN";
    case K4A_CALIBRATION_LENS_DISTORTION_MODEL_THETA:
      return "K4A_CALIBRATION_LENS_DISTORTION_MODEL_THETA";
    case K4A_CALIBRATION_LENS_DISTORTION_MODEL_POLYNOMIAL_3K:
      return "K4A_CALIBRATION_LENS_DISTORTION_MODEL_POLYNOMIAL_3K";
    case K4A_CALIBRATION_LENS_DISTORTION_MODEL_RATIONAL_6KT:
      return "K4A_CALIBRATION_LENS_DISTORTION_MODEL_RATIONAL_6KT";
    case K4A_CALIBRATION_LENS_DISTORTION_MODEL_BROWN_CONRADY:
      return "K4A_CALIBRATION_LENS_DISTORTION_MODEL_BROWN_CONRADY";
    default:
      return "Unknown";
  }
}

static json convert_k4a_calibration_intrinsics(
    k4a_calibration_intrinsics_t intrinsics) {
  json intrinsics_data;
  intrinsics_data[kTypeKeyName] =
      convert_k4a_calibration_model_type_to_string(intrinsics.type);
  intrinsics_data[kParamterCountKeyName] = intrinsics.parameter_count;
  intrinsics_data[kParametersKeyName] =
      convert_k4a_calibration_intrinsic_parameters(intrinsics.parameters);
  return intrinsics_data;
}

static json convert_k4a_calibration_camera(
    k4a_calibration_camera_t camera_calibration) {
  json camera_data;
  camera_data[kExtrinsicsKeyName] =
      convert_k4a_calibration_extrinsics(camera_calibration.extrinsics);
  camera_data[kIntrinsicsKeyName] =
      convert_k4a_calibration_intrinsics(camera_calibration.intrinsics);
  camera_data[kResolutionWidthKeyName] = camera_calibration.resolution_width;
  camera_data[kResolutionHeightKeyName] = camera_calibration.resolution_height;
  camera_data[kMetricRadiusKeyName] = camera_calibration.metric_radius;
  return camera_data;
}

static json convert_k4a_calibration_extrinsics_matrix(
    k4a_calibration_extrinsics_t extrinsics[K4A_CALIBRATION_TYPE_NUM]
                                           [K4A_CALIBRATION_TYPE_NUM]) {
  json extrinsics_matrix_data[K4A_CALIBRATION_TYPE_NUM]
                             [K4A_CALIBRATION_TYPE_NUM];
  for (int i = 0; i < K4A_CALIBRATION_TYPE_NUM; i++) {
    for (int j = 0; j < K4A_CALIBRATION_TYPE_NUM; j++) {
      extrinsics_matrix_data[i][j] =
          convert_k4a_calibration_extrinsics(extrinsics[i][j]);
    }
  }
  return extrinsics_matrix_data;
}

std::string convert_k4a_depth_mode_to_string(k4a_depth_mode_t depth_mode) {
  switch (depth_mode) {
    case K4A_DEPTH_MODE_OFF:
      return "K4A_DEPTH_MODE_OFF";
    case K4A_DEPTH_MODE_NFOV_2X2BINNED:
      return "K4A_DEPTH_MODE_NFOV_2X2BINNED";
    case K4A_DEPTH_MODE_NFOV_UNBINNED:
      return "K4A_DEPTH_MODE_NFOV_UNBINNED ";
    case K4A_DEPTH_MODE_WFOV_2X2BINNED:
      return "K4A_DEPTH_MODE_WFOV_2X2BINNED";
    case K4A_DEPTH_MODE_WFOV_UNBINNED:
      return "K4A_DEPTH_MODE_WFOV_UNBINNED";
    case K4A_DEPTH_MODE_PASSIVE_IR:
      return "K4A_DEPTH_MODE_PASSIVE_IR";
    default:
      return "Unknown";
  }
}

std::string convert_k4a_color_resolution_to_string(
    k4a_color_resolution_t color_mode) {
  switch (color_mode) {
    case K4A_COLOR_RESOLUTION_OFF:
      return "K4A_COLOR_RESOLUTION_OFF";
    case K4A_COLOR_RESOLUTION_720P:
      return "K4A_COLOR_RESOLUTION_720P";
    case K4A_COLOR_RESOLUTION_1080P:
      return "K4A_COLOR_RESOLUTION_1080P";
    case K4A_COLOR_RESOLUTION_1440P:
      return "K4A_COLOR_RESOLUTION_1440P";
    case K4A_COLOR_RESOLUTION_1536P:
      return "K4A_COLOR_RESOLUTION_1536P";
    case K4A_COLOR_RESOLUTION_2160P:
      return "K4A_COLOR_RESOLUTION_2160P";
    case K4A_COLOR_RESOLUTION_3072P:
      return "K4A_COLOR_RESOLUTION_3072P";
    default:
      return "Unknown";
  }
}

static json convert_k4a_calibration(k4a_calibration_t calibration) {
  json calibration_data;

  calibration_data[kDepthCameraCalibrationKeyName] =
      convert_k4a_calibration_camera(calibration.depth_camera_calibration);
  calibration_data[kColorCameraCalibrationKeyName] =
      convert_k4a_calibration_camera(calibration.color_camera_calibration);
  calibration_data[kExtrinsicsKeyName] =
      convert_k4a_calibration_extrinsics_matrix(calibration.extrinsics);
  calibration_data[kDepthModeKeyName] =
      convert_k4a_depth_mode_to_string(calibration.depth_mode);
  // calibration_data[kDepthModeKeyName] = calibration.depth_mode;
  calibration_data[kColorResolutionKeyName] =
      convert_k4a_color_resolution_to_string(calibration.color_resolution);

  return calibration_data;
}

// =================================================================================
// Main function
// =================================================================================

static void clean_up(k4a_playback_t playback) {
  if (playback != NULL) {
    k4a_playback_close(playback);
  }
}

static int playback_cmd_handler(
    char* input_path,
    std::string output_path = "./outputs/calibration_data.json") {
  k4a_playback_t playback = NULL;
  k4a_calibration_t calibration;
  k4a_result_t result;
  json json_data;

  // Open recording
  result = k4a_playback_open(input_path, &playback);
  if (result != K4A_RESULT_SUCCEEDED || playback == NULL) {
    printf("Failed to open recording %s\n", input_path);
    clean_up(playback);
    return 1;
  }

  // Extract calibration data (k4a_calibration_t) from the playback
  if (K4A_RESULT_SUCCEEDED !=
      k4a_playback_get_calibration(playback, &calibration)) {
    printf("Failed to get calibration\n");
    clean_up(playback);
    return 1;
  }

  json_data = convert_k4a_calibration(calibration);

  // Save calibration data to JSON file.
  std::cout << "Save camera paramters to " << output_path << std::endl;
  // Check output directory exists. If not, create it.
  std::filesystem::path dir_path =
      std::filesystem::path(output_path).parent_path();
  if (!std::filesystem::exists(dir_path)) {
    std::cout << "Create directory: " << dir_path << std::endl;
    std::filesystem::create_directories(dir_path);
  }
  std::ofstream output_file_stream(output_path);
  output_file_stream << std::setw(4) << json_data << std::endl;

  clean_up(playback);
  return 0;
}

static void print_usage() {
  printf(
      "Usage: k4a_calibration_data_extractor <filename.mkv> [output_path]\n");
}

int main(int argc, char** argv) {
  int returnCode = 0;

  if (argc == 3) {
    returnCode = playback_cmd_handler(argv[1], argv[2]);
  } else {
    print_usage();
  }

  return returnCode;
}