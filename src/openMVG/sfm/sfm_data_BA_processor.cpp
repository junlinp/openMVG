#include "sfm_data_BA_processor.hpp"

#include "ceres/problem.h"

ceres::CostFunction* IntrinsicsToCostFunction(
    openMVG::cameras::EINTRINSIC intrinsic, const openMVG::Vec2& observation,
    const double weight) {
  switch (intrinsic) {
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
    // case openMVG::cameras::EINTRINSIC::CAMERA_SPHERICAL:
    //  return openMVG::sfm::ResidualErrorFunctor_Intrinsic_Spherical::Create(
    //     intrinsic, observation, weight);
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
    double d = 1.0 / rho;
    residual[0] = T(0.0);
    for (int i = 0; i < vector_size; i++) {
      residual[0] += T(y[i]) * (x[i] - T(z[i])) +
                     d * (x[i] - T(z[i])) * (x[i] - T(z[i]));
    }
    return true;
  }
};

void CPUProcessor::OptimizeParameters() {
  /*
  std::cout << "Origin Camera : " << std::endl;
  for (auto& camera_para : camera_params_) {
      std::cout << camera_para << " ";
  }
  std::cout << std::endl;

  std::cout << "Origin pose : " << std::endl;
  for (auto& camera_para : pose_params_) {
      std::cout << camera_para << " ";
  }
  std::cout << std::endl;

  std::cout << "Origin structure : " << std::endl;
  for (auto& camera_para : structure_params_) {
      std::cout << camera_para << " ";
  }
  std::cout << std::endl;
  */
  ceres::CostFunction* intrins_cost_function = IntrinsicsToCostFunction(
      camera_type_, Eigen::Map<const Eigen::Vector2d>(&observations_[0]),
      structure_weight_);
  ceres::Problem problem;

  problem.AddParameterBlock(&camera_params_[0], camera_params_.size());
  ceres::CostFunction* pose_multipler_cost =
      new ceres::AutoDiffCostFunction<MultiplerCost, 1, 6>(new MultiplerCost(
          pose_params_multipler_, pose_consensus_params_, rho_));
  ceres::CostFunction* structure_multipler_cost =
      new ceres::AutoDiffCostFunction<MultiplerCost, 1, 3>(new MultiplerCost(
          structure_params_multipler_, structure_consensus_params_, rho_));

  if (options_.intrinsics_opt !=
      openMVG::cameras::Intrinsic_Parameter_Type::ADJUST_ALL) {
    problem.SetParameterBlockConstant(&camera_params_[0]);
  }

  problem.AddParameterBlock(&pose_params_[0], pose_params_.size());
  if (options_.extrinsics_opt !=
      openMVG::sfm::Extrinsic_Parameter_Type::ADJUST_ALL) {
    problem.SetParameterBlockConstant(&pose_params_[0]);
  }

  problem.AddParameterBlock(&structure_params_[0], structure_params_.size());
  if (options_.structure_opt !=
      openMVG::sfm::Structure_Parameter_Type::ADJUST_ALL) {
    problem.SetParameterBlockConstant(&structure_params_[0]);
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
  /*
  std::cout << "Camera : " << std::endl;
  for (auto& camera_para : camera_params_) {
      std::cout << camera_para << " ";
  }
  std::cout << std::endl;

  std::cout << "pose : " << std::endl;
  for (auto& camera_para : pose_params_) {
      std::cout << camera_para << " ";
  }
  std::cout << std::endl;

  std::cout << "structure : " << std::endl;
  for (auto& camera_para : structure_params_) {
      std::cout << camera_para << " ";
  }
  std::cout << std::endl;
  */
}
}  // namespace sfm
}  // namespace openMVG
