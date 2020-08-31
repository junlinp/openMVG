#include "sfm_data_BA_processor.hpp"

#include "ceres/problem.h"
#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Camera_Intrinsics.hpp"

ceres::CostFunction* IntrinsicsToCostFunction(
    openMVG::cameras::IntrinsicBase* intrinsic,
    const openMVG::Vec2& observation, const double weight) {
  switch (intrinsic->getType()) {
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA:
      return openMVG::sfm::ResidualErrorFunctor_Pinhole_Intrinsic::Create(
          observation, weight);
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_RADIAL1:
      return openMVG::sfm::ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K1::
          Create(observation, weight);
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_RADIAL3:
      return openMVG::sfm::ResidualErrorFunctor_Pinhole_Intrinsic_Radial_K3::
          Create(observation, weight);
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_BROWN:
      return openMVG::sfm::ResidualErrorFunctor_Pinhole_Intrinsic_Brown_T2::
          Create(observation, weight);
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_FISHEYE:
      return openMVG::sfm::ResidualErrorFunctor_Pinhole_Intrinsic_Fisheye::
          Create(observation, weight);
    case openMVG::cameras::EINTRINSIC::CAMERA_SPHERICAL:
      return openMVG::sfm::ResidualErrorFunctor_Intrinsic_Spherical::Create(
          intrinsic, observation, weight);
    default:
      return {};
  }
}

namespace openMVG {
namespace sfm {

struct MultiplerCost {
  std::vector<double> y, z;
  double rho;

  MultiplerCost(std::vector<double> y, std::vector<double> z, double rho_) {
    this->y = y;
    this->z = z;
    rho = rho_;
  }

  template <class T>
  bool operator()(const T* x, T* residual) const {
    const size_t vector_size = y.size();
    double d = rho / 2.0;
    residual[0] = T(0.0);
    for (int i = 0; i < vector_size; i++) {
      residual[0] +=
          T(y[i]) * (x[i] - T(z[i])) + d * (x[i] - T(z[i])) * (x[i] - T(z[i]));
    }
    return true;
  }
};

void CPUProcessor::XOptimization() {
  ceres::CostFunction* intrins_cost_function = IntrinsicsToCostFunction(
      camera_ptr_, Eigen::Vector2d(observations_[0], observations_[1]),
      structure_weight_);
  ceres::Problem problem;

  problem.AddParameterBlock(&camera_params_[0], camera_params_.size());
  ceres::CostFunction* pose_multipler_cost =
      new ceres::AutoDiffCostFunction<MultiplerCost, 1, 6>(new MultiplerCost(
          pose_params_multipler_, pose_consensus_params_, rho_));
  ceres::CostFunction* structure_multipler_cost =
      new ceres::AutoDiffCostFunction<MultiplerCost, 1, 3>(new MultiplerCost(
          structure_params_multipler_, structure_consensus_params_, rho_));

  if (options_.intrinsics_opt ==
      openMVG::cameras::Intrinsic_Parameter_Type::NONE) {
    problem.SetParameterBlockConstant(&camera_params_[0]);
    camera_consensus_params_ = camera_params_;
  } else {
    const std::vector<int> vec_constant_intrinsic =
        camera_ptr_->subsetParameterization(options_.intrinsics_opt);
    if (!vec_constant_intrinsic.empty()) {
      ceres::SubsetParameterization* subset_parameterization =
          new ceres::SubsetParameterization(camera_params_.size(),
                                            vec_constant_intrinsic);
      problem.SetParameterization(&camera_params_[0], subset_parameterization);
    }
  }

  problem.AddParameterBlock(&pose_params_[0], pose_params_.size());
  // Pose SubParameterization
  if (options_.extrinsics_opt == openMVG::sfm::Extrinsic_Parameter_Type::NONE) {
    problem.SetParameterBlockConstant(&pose_params_[0]);
    pose_consensus_params_ = pose_params_;
  } else {  // Subset parametrization
    std::vector<int> vec_constant_extrinsic;
    if (options_.extrinsics_opt ==
        openMVG::sfm::Extrinsic_Parameter_Type::ADJUST_ROTATION) {
      vec_constant_extrinsic.insert(vec_constant_extrinsic.end(), {0, 1, 2});
    }
    if (options_.extrinsics_opt ==
        openMVG::sfm::Extrinsic_Parameter_Type::ADJUST_TRANSLATION) {
      vec_constant_extrinsic.insert(vec_constant_extrinsic.end(), {3, 4, 5});
    }

    if (!vec_constant_extrinsic.empty()) {
      ceres::SubsetParameterization* subset_parameterization =
          new ceres::SubsetParameterization(6, vec_constant_extrinsic);
      problem.SetParameterization(&pose_params_[0], subset_parameterization);
    }
  }

  problem.AddParameterBlock(&structure_params_[0], structure_params_.size());
  if (options_.structure_opt == openMVG::sfm::Structure_Parameter_Type::NONE) {
    problem.SetParameterBlockConstant(&structure_params_[0]);
    structure_consensus_params_ = structure_params_;
  }
  problem.AddResidualBlock(intrins_cost_function, nullptr, &camera_params_[0],
                           &pose_params_[0], &structure_params_[0]);

  problem.AddResidualBlock(pose_multipler_cost, nullptr, &pose_params_[0]);

  problem.AddResidualBlock(structure_multipler_cost, nullptr,
                           &structure_params_[0]);

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
}

void CPUProcessor::ZOptimization() {
  camera_consensus_params_ = ZUpdate(
      camera_params_, camera_params_multipler_,
      camera_ptr_->subsetParameterization(options_.intrinsics_opt), rho_);
  pose_consensus_params_ =
      ZUpdate(pose_consensus_params_, pose_params_multipler_,
              PoseSubParameterization(options_.extrinsics_opt), rho_);

  structure_consensus_params_ =
      ZUpdate(structure_consensus_params_, structure_params_multipler_,
              StructureSubParameterization(options_.structure_opt), rho_);
}

void CPUProcessor::YUpdate() {
  for (size_t i = 0; i < camera_params_multipler_.size(); i++) {
    camera_params_multipler_[i] +=
        rho_ * (camera_params_[i] - camera_consensus_params_[i]);
  }

  for (size_t i = 0; i < pose_params_multipler_.size(); i++) {
    pose_params_multipler_[i] +=
        rho_ * (pose_params_[i] - pose_consensus_params_[i]);
  }

  for (size_t i = 0; i < structure_params_multipler_.size(); i++) {
    structure_params_multipler_[i] +=
        rho_ * (structure_params_[i] - structure_consensus_params_[i]);
  }
}

std::vector<double> CPUProcessor::getLocalCameraParams() {
  return camera_consensus_params_;
};

std::vector<double> CPUProcessor::getLocalPoseParams() {
  return pose_consensus_params_;
};

std::vector<double> CPUProcessor::getLocalStructureParams() {
  return structure_consensus_params_;
};
void CPUProcessor::setUpdateCameraParamsConsusence(
    const std::vector<double>& camera_consensus_params) {
  assert(camera_consensus_params.size() == camera_consensus_params_.size());
  camera_consensus_params_ = camera_consensus_params;
}
void CPUProcessor::setUpdatePoseParamsConsusence(
    const std::vector<double>& pose_consensus_params) {
  assert(pose_consensus_params.size() == pose_consensus_params_.size());
  pose_consensus_params_ = pose_consensus_params;
}
void CPUProcessor::setUpdateStructureParams(
    const std::vector<double>& structure_consensus_params) {
  assert(structure_consensus_params.size() ==
         structure_consensus_params_.size());
  structure_consensus_params_ = structure_consensus_params;
  std::vector<double> delta_y =
      YUpdate(structure_params_, structure_consensus_params_, rho_);
  for (int i = 0; i < structure_params_multipler_.size(); i++) {
    structure_params_multipler_[i] += delta_y[i];
  }
}

}  // namespace sfm
}  // namespace openMVG