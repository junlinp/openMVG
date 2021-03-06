// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

// Copyright (c) 2020 panjunlin Pierre Moulon.

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "sfm_data_BA_admm.hpp"

#include "ceres/rotation.h"
#include "openMVG/geometry/Similarity3.hpp"
#include "openMVG/geometry/Similarity3_Kernel.hpp"
#include "openMVG/robust_estimation/robust_estimator_LMeds.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_transform.hpp"
#include "sfm_data_BA_processor.hpp"
#ifdef OPENMVG_USE_OPENMP
#include <omp.h>
#endif
namespace openMVG {
namespace sfm {

double ApplyGPS(SfM_Data& sfm_data, const Optimize_Options& options) {
  double pose_center_robust_fitting_error = 0.0;
  openMVG::geometry::Similarity3 sim_to_center;
  bool b_usable_prior = false;
  if (options.use_motion_priors_opt && sfm_data.GetViews().size() > 3) {
    // - Compute a robust X-Y affine transformation & apply it
    // - This early transformation enhance the conditionning (solution closer to
    // the Prior coordinate system)
    {
      // Collect corresponding camera centers
      std::vector<Vec3> X_SfM, X_GPS;
      for (const auto& view_it : sfm_data.GetViews()) {
        const sfm::ViewPriors* prior =
            dynamic_cast<sfm::ViewPriors*>(view_it.second.get());
        if (prior != nullptr && prior->b_use_pose_center_ &&
            sfm_data.IsPoseAndIntrinsicDefined(prior)) {
          X_SfM.push_back(sfm_data.GetPoses().at(prior->id_pose).center());
          X_GPS.push_back(prior->pose_center_);
        }
      }
      openMVG::geometry::Similarity3 sim;

      // Compute the registration:
      if (X_GPS.size() > 3) {
        const Mat X_SfM_Mat = Eigen::Map<Mat>(X_SfM[0].data(), 3, X_SfM.size());
        const Mat X_GPS_Mat = Eigen::Map<Mat>(X_GPS[0].data(), 3, X_GPS.size());
        geometry::kernel::Similarity3_Kernel kernel(X_SfM_Mat, X_GPS_Mat);
        const double lmeds_median =
            openMVG::robust::LeastMedianOfSquares(kernel, &sim);
        if (lmeds_median != std::numeric_limits<double>::max()) {
          b_usable_prior = true;  // PRIOR can be used safely

          // Compute the median residual error once the registration is applied
          for (Vec3& pos :
               X_SfM)  // Transform SfM poses for residual computation
          {
            pos = sim(pos);
          }
          Vec residual = (Eigen::Map<Mat3X>(X_SfM[0].data(), 3, X_SfM.size()) -
                          Eigen::Map<Mat3X>(X_GPS[0].data(), 3, X_GPS.size()))
                             .colwise()
                             .norm();
          std::sort(residual.data(), residual.data() + residual.size());
          pose_center_robust_fitting_error = residual(residual.size() / 2);

          // Apply the found transformation to the SfM Data Scene
          openMVG::sfm::ApplySimilarity(sim, sfm_data);

          // Move entire scene to center for better numerical stability
          Vec3 pose_centroid = Vec3::Zero();
          for (const auto& pose_it : sfm_data.poses) {
            pose_centroid +=
                (pose_it.second.center() / (double)sfm_data.poses.size());
          }
          sim_to_center = openMVG::geometry::Similarity3(
              openMVG::sfm::Pose3(Mat3::Identity(), pose_centroid), 1.0);
          openMVG::sfm::ApplySimilarity(sim_to_center, sfm_data, true);
        }
      }
    }
  }
  return pose_center_robust_fitting_error;
}
bool Bundle_Adjustment_Admm::Adjust(SfM_Data& sfm_data,
                                    const Optimize_Options& options) {
  ApplyGPS(sfm_data, options);
  std::map<openMVG::IndexT, std::shared_ptr<Processor>> view_id_map_processor;
  std::map<openMVG::IndexT, std::shared_ptr<Processor>> pose_id_map_processor;
  std::map<openMVG::IndexT, std::shared_ptr<Processor>>
      structure_id_map_processor;
  std::vector<std::shared_ptr<Processor>> total_processors;

  // initialize the global consensus variable
  std::map<IndexT, std::vector<double>> map_intrinsics;
  std::map<IndexT, std::vector<double>> map_poses;
  std::map<IndexT, std::vector<double>> map_structure;
  std::map<IndexT, std::vector<double>> map_control_point;

  // Setup Intrinsics data
  for (const auto& intrinsic_it : sfm_data.intrinsics) {
    const IndexT index_cam = intrinsic_it.first;
    if (isValid(intrinsic_it.second->getType())) {
      map_intrinsics[index_cam] = intrinsic_it.second->getParams();
    } else {
      std::cerr << "Unsupported camera type." << std::endl;
    }
  }

  // Setup Poses data
  for (const auto& pose_it : sfm_data.poses) {
    const IndexT index_pose = pose_it.first;
    const Pose3& pose = pose_it.second;
    const Mat3 R = pose.rotation();
    const Vec3 t = pose.translation();
    double angleAxis[3];
    ceres::RotationMatrixToAngleAxis((const double*)R.data(), angleAxis);
    map_poses[index_pose] = {angleAxis[0], angleAxis[1], angleAxis[2],
                             t(0),         t(1),         t(2)};
  }

  for (const auto& structure_landmark_it : sfm_data.structure) {
    const Observations& obs = structure_landmark_it.second.obs;
    const IndexT id_structure = structure_landmark_it.first;
    const double* X = structure_landmark_it.second.X.data();
    map_structure[id_structure] = {X[0], X[1], X[2]};

    for (const auto& obs_it : obs) {
      const View* view = sfm_data.views.at(obs_it.first).get();
      const auto intrinsic = sfm_data.intrinsics.at(view->id_intrinsic);
      std::vector<double> ob_x = {obs_it.second.x(0), obs_it.second.x(1)};
      auto processor = std::make_shared<CPUProcessor>(
          intrinsic.get(), map_intrinsics[view->id_intrinsic],
          map_poses[view->id_pose], map_structure[id_structure], ob_x, options);

      view_id_map_processor[view->id_view] = processor;
      pose_id_map_processor[view->id_pose] = processor;
      structure_id_map_processor[id_structure] = processor;
      total_processors.push_back(processor);
    }
  }

  // one epoch
  //
  // std::for_each(total_processors.begin(), total_processors.end(),
  //              [](const auto& element) { element->X_update(); });

  // collect all the variable
  std::vector<double> initial_;
  // TODO:
  // this binary function can reuse;
  // std::accumulate(view_id_map_processor.begin(), view_id_map_processor.end(),
  //                initial_,
  //                [](const std::vector<double>& init, const auto& item) {
  //                  auto res = init;
  //                  if (init.size() == 0) {
  //                    res = item->getLocalCameraParams();
  //                  } else {
  //                    res = init + item->getLocalCameraParams();
  //                  }
  //                  return res;
  //                });
  return false;
}
}  // namespace sfm
}  // namespace openMVG