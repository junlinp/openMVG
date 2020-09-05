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
    T d = T(rho) / T(2.0);
    residual[0] = T(0.0);
    T quadratic(0.0);
    for (int i = 0; i < vector_size; i++) {
      residual[0] += T(y[i]) * (x[i] - T(z[i]));
      quadratic += (x[i] - T(z[i])) * (x[i] - T(z[i]));
    }
    residual[0] += quadratic * d;
    return true;
  }
};

ceres::CostFunction* IntrinsicsToMultiplerCost(
    openMVG::cameras::IntrinsicBase* intrinsic, std::vector<double> multipler,
    std::vector<double> consensus_paramer, double rho) {
  switch (intrinsic->getType()) {
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA:
      // 3
      return new ceres::AutoDiffCostFunction<MultiplerCost, 1, 3>(
          new MultiplerCost(multipler, consensus_paramer, rho));
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_RADIAL1:
      // 4
      return new ceres::AutoDiffCostFunction<MultiplerCost, 1, 4>(
          new MultiplerCost(multipler, consensus_paramer, rho));
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_RADIAL3:
      // 6
      return new ceres::AutoDiffCostFunction<MultiplerCost, 1, 6>(
          new MultiplerCost(multipler, consensus_paramer, rho));

    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_BROWN:
      // 8
      return new ceres::AutoDiffCostFunction<MultiplerCost, 1, 8>(
          new MultiplerCost(multipler, consensus_paramer, rho));
    case openMVG::cameras::EINTRINSIC::PINHOLE_CAMERA_FISHEYE:
      // 7
      return new ceres::AutoDiffCostFunction<MultiplerCost, 1, 7>(
          new MultiplerCost(multipler, consensus_paramer, rho));
    /*
    case openMVG::cameras::EINTRINSIC::CAMERA_SPHERICAL:
      return openMVG::sfm::ResidualErrorFunctor_Intrinsic_Spherical::Create(
          intrinsic, observation, weight);
    */
    default:
      return {};
  }
}

double CPUProcessor::XOptimization() {
  ceres::CostFunction* intrins_cost_function = IntrinsicsToCostFunction(
      camera_ptr_, Eigen::Vector2d(observations_[0], observations_[1]),
      structure_weight_);
  ceres::Problem problem;

  problem.AddParameterBlock(&camera_params_[0], camera_params_.size());
  // TODO: camera_multipler_cost
  ceres::CostFunction* camera_multipler_cost = IntrinsicsToMultiplerCost(
      camera_ptr_, camera_params_multipler_, camera_consensus_params_, rho_);

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

  problem.AddResidualBlock(camera_multipler_cost, nullptr, &camera_params_[0]);

  problem.AddResidualBlock(pose_multipler_cost, nullptr, &pose_params_[0]);

  problem.AddResidualBlock(structure_multipler_cost, nullptr,
                           &structure_params_[0]);

  ceres::Solver::Options options;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  // std::cout << summary.BriefReport() << std::endl;
  double cost = 0.0;
  problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
  return cost;
}

void CPUProcessor::ZOptimization() {
  dual_residual_square_normal_ = 0.0;
  std::vector<double> last_camera_consensus_params = camera_consensus_params_;
  camera_consensus_params_ = ZUpdate(
      camera_params_, camera_params_multipler_,
      camera_ptr_->subsetParameterization(options_.intrinsics_opt), rho_);
  for (int i = 0; i < camera_consensus_params_.size(); i++) {
    double d = camera_consensus_params_[i] - last_camera_consensus_params[i];
    dual_residual_square_normal_ += d * d;
  }

  std::vector<double> last_pose_consensus_params = pose_consensus_params_;
  pose_consensus_params_ =
      ZUpdate(pose_consensus_params_, pose_params_multipler_,
              PoseSubParameterization(options_.extrinsics_opt), rho_);
  for (int i = 0; i < pose_consensus_params_.size(); i++) {
    double d = pose_consensus_params_[i] - last_pose_consensus_params[i];
    dual_residual_square_normal_ += d * d;
  }

  std::vector<double> last_structure_consensus_params =
      structure_consensus_params_;
  structure_consensus_params_ =
      ZUpdate(structure_consensus_params_, structure_params_multipler_,
              StructureSubParameterization(options_.structure_opt), rho_);
  for (int i = 0; i < structure_consensus_params_.size(); i++) {
    double d =
        structure_consensus_params_[i] - last_structure_consensus_params[i];
    dual_residual_square_normal_ += d * d;
  }

  dual_residual_square_normal_ *= rho_ * rho_;
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

double CPUProcessor::PrimaryResidual() {
  double residual_square_normal = 0.0;

  for (int i = 0; i < camera_params_.size(); i++) {
    double d = camera_params_[i] - camera_consensus_params_[i];
    residual_square_normal += d * d;
  }

  for (int i = 0; i < pose_params_.size(); i++) {
    double d = pose_params_[i] - pose_consensus_params_[i];
    residual_square_normal += d * d;
  }

  for (int i = 0; i < structure_params_.size(); i++) {
    double d = structure_params_[i] - structure_consensus_params_[i];
    residual_square_normal += d * d;
  }
  return residual_square_normal;
}

double CPUProcessor::DualResidual() { return dual_residual_square_normal_; }

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
}

}  // namespace sfm
}  // namespace openMVG