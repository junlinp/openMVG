#ifndef SFM_DATA_BA_PROCESSOR_HPP_
#define SFM_DATA_BA_PROCESSOR_HPP_
#include "openMVG/cameras/Camera_Common.hpp"
#include "openMVG/cameras/Camera_Intrinsics.hpp"
#include "openMVG/cameras/Camera_Spherical.hpp"
#include "sfm_data_BA.hpp"
#include "sfm_data_BA_ceres_camera_functor.hpp"
namespace openMVG {
namespace sfm {
class Processor {
 public:
  Processor(openMVG::cameras::IntrinsicBase* camera_ptr,
            const std::vector<double>& camera_params,
            const std::vector<double>& pose_params,
            const std::vector<double>& structure_params,
            const std::vector<double>& observation,
            const Optimize_Options& options,
            const double structure_weight = 0.0, const double rho = 1.0)
      : camera_ptr_(camera_ptr),
        camera_params_(camera_params),
        pose_params_(pose_params),
        structure_params_(structure_params),
        observations_(observation),
        options_(options),
        structure_weight_(structure_weight),
        rho_(rho),
        camera_params_multipler_(camera_params.size(), 0.0),
        pose_params_multipler_(pose_params.size(), 0.0),
        structure_params_multipler_(structure_params.size(), 0.0),
        camera_consensus_params_(camera_params),
        pose_consensus_params_(pose_params),
        structure_consensus_params_(structure_params) {}
  virtual ~Processor() {}
  virtual void XOptimization() = 0;
  // params + 1.0 / rho * params_multipler
  virtual void ZOptimization() = 0;
  virtual void YUpdate() = 0;

  virtual std::vector<double> getLocalCameraParams() = 0;
  virtual std::vector<double> getLocalPoseParams() = 0;
  virtual std::vector<double> getLocalStructureParams() = 0;

  virtual void setUpdateCameraParamsConsusence(
      const std::vector<double>& camera_params) {}
  virtual void setUpdatePoseParamsConsusence(
      const std::vector<double>& pose_params) {}
  virtual void setUpdateStructureParams(
      const std::vector<double>& structure_params) {}

 protected:
  openMVG::cameras::IntrinsicBase* camera_ptr_;
  std::vector<double> camera_params_;
  std::vector<double> pose_params_;
  std::vector<double> structure_params_;
  std::vector<double> observations_;
  std::vector<double> camera_params_multipler_;
  std::vector<double> pose_params_multipler_;
  std::vector<double> structure_params_multipler_;
  std::vector<double> camera_consensus_params_;  // z
  std::vector<double> pose_consensus_params_;
  std::vector<double> structure_consensus_params_;

  Optimize_Options options_;
  double structure_weight_;
  double rho_;
};

class CPUProcessor : public Processor {
 public:
  CPUProcessor(openMVG::cameras::IntrinsicBase* camera_ptr_,
               const std::vector<double>& camera_params,
               const std::vector<double>& pose_params,
               const std::vector<double>& structure_params,
               const std::vector<double>& observation,
               const Optimize_Options& options,
               const double structure_weight = 0.0, const double rho = 1.0)
      : Processor(camera_ptr_, camera_params, pose_params, structure_params,
                  observation, options, structure_weight, rho) {}

  void XOptimization() override;
  // params + 1.0 / rho * params_multipler
  void ZOptimization() override;
  void YUpdate() override;

  std::vector<double> getLocalCameraParams() override;
  std::vector<double> getLocalPoseParams() override;
  std::vector<double> getLocalStructureParams() override;

  void setUpdateCameraParamsConsusence(
      const std::vector<double>& camera_consensus_params) override;

  void setUpdatePoseParamsConsusence(
      const std::vector<double>& pose_consensus_params) override;

  void setUpdateStructureParams(
      const std::vector<double>& structure_consensus_params) override;

 private:
 std::vector<int> PoseSubParameterization(openMVG::sfm::Extrinsic_Parameter_Type pose_subparameterization_option) {
   std::vector<int> res;
   if (pose_subparameterization_option == openMVG::sfm::Extrinsic_Parameter_Type::ADJUST_ROTATION) {
     res = {0, 1, 2};
   } else if (pose_subparameterization_option == openMVG::sfm::Extrinsic_Parameter_Type::ADJUST_TRANSLATION) {
     res = {3, 4, 5};
   } else if (pose_subparameterization_option == openMVG::sfm::Extrinsic_Parameter_Type::NONE) {
     res = {0, 1, 2, 3, 4, 5};
   }
   return res;
 }

 std::vector<int> StructureSubParameterization(openMVG::sfm::Structure_Parameter_Type structure_subparameterization_option) {
   std::vector<int> res;
   if (structure_subparameterization_option == openMVG::sfm::Structure_Parameter_Type::NONE) {
     res = {0, 1, 2};
   }
   return res;
 } 
  std::vector<double> ZUpdate(std::vector<double> origin_params,
                              std::vector<double> origin_params_multipler,
                              const std::vector<int>& vec_constant_index, double rho) {
    std::set<int> s;
    s.insert(begin(vec_constant_index), end(vec_constant_index));

    double d = 1.0 / (rho_ + std::numeric_limits<double>::epsilon());
    std::vector<double> res_vector;
    for (int i = 0; i < origin_params.size(); i++) {
      if (s.find(i) != s.end()) {
        res_vector.push_back(origin_params[i]);
      } else {
        res_vector.push_back(origin_params[i] - d * origin_params_multipler[i]);
      }
    }
    return res_vector;
  }

  std::vector<double> YUpdate(std::vector<double>& origin_params,
                              std::vector<double>& origin_consusens_params,
                              double rho) {
    std::vector<double> res;
    for (int i = 0; i < origin_params.size(); i++) {
      res.push_back(rho * (origin_params[i] - origin_consusens_params[i]));
    }
    return res;
  }
};

}  // namespace sfm

}  // namespace openMVG
#endif  // SFM_DATA_BA_PROCESSOR_HPP_