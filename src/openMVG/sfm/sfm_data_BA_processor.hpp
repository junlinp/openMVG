#ifndef SFM_DATA_BA_PROCESSOR_HPP_
#define SFM_DATA_BA_PROCESSOR_HPP_
#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "sfm_data_BA.hpp"
#include "sfm_data_BA_ceres_camera_functor.hpp"
ceres::CostFunction* IntrinsicsToCostFunction(
    openMVG::cameras::EINTRINSIC intrinsic, const Vec2& observation,
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
    case openMVG::cameras::EINTRINSIC::CAMERA_SPHERICAL:
      return openMVG::sfm::ResidualErrorFunctor_Intrinsic_Spherical::Create(
          intrinsic, observation, weight);
    default:
      return {};
  }
}
namespace openMVG {
namespace sfm {
class Processor {
 public:
  Processor(openMVG::cameras::EINTRINSIC camera_type,
            const std::vector<double>& camera_params,
            const std::vector<double>& pose_params,
            const std::vector<double>& structure_params,
            const Optimize_Options& options,
            const double structure_weight = 0.0, const double rho = 1.0)
      : camera_type_(camera_type),
        camera_params_(camera_params),
        pose_params_(pose_params),
        structure_params_(structure_params),
        options_(options),
        structure_weight_(structure_weight),
        rho_(rho) {}
  virtual void OptimizeParameters() = 0;
  // TODO: junlinp@qq.com
  // return params + 1.0 / rho * params_multipler
  virtual std::vector<double> getLocalCameraParams();
  virtual std::vector<double> getLocalPoseParams();
  virtual std::vector<double> getLocalStructureParams();

  virtual void setUpdateCameraParamsConsusence(
      const std::vector<double>& camera_params);
  virtual void setUpdatePoseParamsConsusence(
      const std::vector<double>& pose_params);
  virtual void setUpdateStructureParams(
      const std::vector<double>& structure_params);

 protected:
  openMVG::cameras::EINTRINSIC camera_type_;
  std::vector<double> camera_params_;
  std::vector<double> pose_params_;
  std::vector<double> structure_params_;
  std::vector<double> camera_params_multipler;
  std::vector<double> pose_params_multipler;
  std::vector<double> structure_params_multipler;
  Optimize_Options options_;
  double structure_weight_;
  double rho_;
};

class CPUProcessor : public Processor {
  CPUProcessor(openMVG::cameras::EINTRINSIC camera_type,
               const std::vector<double>& camera_params,
               const std::vector<double>& pose_params,
               const std::vector<double>& structure_params,
               const Optimize_Options& options,
               const double structure_weight = 0.0, const double rho = 1.0)
      : Processor(camera_type, camera_params, pose_params, structure_params,
                  options, structure_weight, rho) {}

  void OptimizeParameters() override;
};

}  // namespace sfm

}  // namespace openMVG
#endif  // SFM_DATA_BA_PROCESSOR_HPP_